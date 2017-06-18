/**
 * fractgl v0.1
 * (c) 2017 Brian Nenninger
 * Released under the GNU General Public License version 3.
 */
(() => {
class Complex {
    constructor(real, imag) {
        this.realStr = String(real).trim();
        this.imagStr = String(imag).trim();
        this.real = window.parseFloat(this.realStr);
        if (window.isNaN(this.real)) {
            throw Error('Invalid real part: ' + this.realStr);
        }
        this.imag = window.parseFloat(this.imagStr);
        if (window.isNaN(this.imag)) {
            throw Error('Invalid imaginary part: ' + this.imagStr);
        }
    }

    static fromString(s) {
        s = s.trim();
        const realOnlyMatch = s.match(/^[+-]?[\d.]+$/);
        if (realOnlyMatch) {
            return new Complex(s, 0);
        }
        const imagOnlyMatch = s.match(/^[+-]?[\d.]+i$/);
        if (imagOnlyMatch) {
            return new Complex(0, s.substring(0, s.indexOf('i')));
        }
        const bothMatch = s.match(/^([+-]?[\d.]+)\s*([+-])\s*([\d.]+i)$/);
        if (bothMatch) {
            const imag = bothMatch[2] + bothMatch[3].substring(0, bothMatch[3].indexOf('i'));
            return new Complex(bothMatch[1], imag);
        }
        throw Error('Failed to parse complex number: ' + s);
    }

    static fromArray(arr) {
        return new Complex(arr[0], arr[1] || 0);
    }

    static from(arg) {
        if (Array.isArray(arg)) {
            return Complex.fromArray(arg);
        }
        if (typeof(arg) === 'string') {
            return Complex.fromString(arg);
        }
        if (typeof(arg) === 'number') {
            return new Complex(arg, 0);
        }
        throw Error('Invalid argument: ' + arg);
    }

    isRealOnly() {
        return this.imag === 0;
    }

    isRealNonnegativeInteger() {
        return this.imag === 0 && (this.real >= 0 && this.real % 1 === 0);
    }

    weightedMean(other, weight) {
        return new Complex(weightedMean(this.real, other.real, weight),
                           weightedMean(this.imag, other.imag, weight));
    }

    equals(other) {
        return this.real === other.real && this.imag === other.imag;
    }

    toString() {
        if (this.imag === 0) {
            return this.realStr;
        }
        if (this.real === 0) {
            return `${this.imagStr}i`;
        }
        const fi = this.imagStr[0];
        const operator = (fi === '-') ? '-' : '+';
        const imagWithoutSign = this.imagStr.substring((fi === '+' || fi === '-') ? 1 : 0);
        return `${this.realStr} ${operator} ${imagWithoutSign}i`;
    }

    toRoundedString(decimalPlaces) {
        const rs = roundToPlaces(this.real, decimalPlaces);
        const is = roundToPlaces(this.imag, decimalPlaces);
        if (this.imag === 0) {
            return rs;
        }
        if (this.real === 0) {
            return `${is}i`;
        }
        const fi = is[0];
        const operator = (fi === '-') ? '-' : '+';
        const imagWithoutSign = is.substring((fi === '+' || fi === '-') ? 1 : 0);
        return `${rs} ${operator} ${imagWithoutSign}i`;
    }
}

Complex.ZERO = new Complex(0, 0);

class Term {
    constructor(coefficient, power) {
        this.coefficient = coefficient;
        this.power = power;
    }

    static fromArray(arr) {
        return new Term(Complex.from(arr[0]), Complex.from(arr[1]));
    }

    weightedMean(other, weight) {
        return new Term(this.coefficient.weightedMean(other.coefficient, weight),
                        this.power.weightedMean(other.power, weight));
    }

    toJson() {
        return [this.coefficient.toString(), this.power.toString()];
    }
}

Term.ZERO = new Term(Complex.ZERO, Complex.ZERO);

class Expression {
    constructor(terms) {
        this.terms = terms;
    }

    static fromArray(arr) {
        return new Expression(arr.map(Term.fromArray));
    }

    static fromString(s) {
        return Expression.fromArray(JSON.parse(s));
    }

    toGlShaderCode(xVar='x', yVar='y', newXVar='newx', newYVar='newy', cxVar='cx', cyVar='cy') {
        // For integer powers we only need multiplications. Except that's slower than the general
        // approach for large powers because of the number of factors. Caching intermediate powers
        // might help (e.g. 'float x2=x*x; float x3=x2*x; ...').
        if (this.terms.every(t =>
                t.coefficient.isRealOnly() && t.power.isRealNonnegativeInteger() && t.power <= 8)) {
            const integerPoly = this.terms.map(t => [t.coefficient.real, t.power.real]);
            const [realParts, imagParts] = expandIntegerPolynomial(integerPoly);
            return `
                float ${newXVar} = ${integerPowersToGlExpr(realParts)} + ${cxVar};
                float ${newYVar} = ${integerPowersToGlExpr(imagParts)} + ${cyVar};
            `;
        }
        // ln(x+yi) = s + ti
        const localVarCode = [
            'float _s = 0.5*log(x*x + y*y);',
            'float _t = atan(y, x);'
        ];

        const xTerms = [];
        const yTerms = [];
        // Each term has its own intermediate variables using ${i} as a suffix.
        let i = 0;
        for (const term of this.terms) {
            if (term.coefficient.isRealOnly() && term.power.isRealOnly()) {
                // a*(x+yi)**c = a*exp(c*ln(x+yi)) = a(exp(cs + i*ct))
                //             = a*(x+yi)**c = a*exp(cs)*(cos(ct) + i*sin(ct))
                // (where s+ti=ln(x+yi) as above)
                const a = numberWithDecimalPoint(term.coefficient.real);
                const c = numberWithDecimalPoint(term.power.real);
                localVarCode.push(`float _mag${i} = ${a}*exp(${c}*_s);`);
                localVarCode.push(`float _phase${i} = ${c}*_t;`)
                xTerms.push(`_mag${i}*cos(_phase${i})`);
                yTerms.push(`_mag${i}*sin(_phase${i})`);
            } else {
                // (a+bi)*(x+yi)**(c+di) = (a+bi)*exp((c+di)*ln(x+yi)) = (a+bi)*exp((c+di)*(s+ti))
                //                       = (a+bi)*exp((cs-dt) + i*(ct+ds))
                //                       = (a+bi)*exp(cs-dt)*(cos(ct+ds) + i*sin(ct+ds))
                // (let m=exp(cs-dt), p=(ct+ds))
                //                       = (a+bi)*m*(cos(p) + i*sin(p))
                //                       = m*(a*cos(p) - b*sin(p)) + i*(a*sin(p) + b*cos(p))
                const a = numberWithDecimalPoint(term.coefficient.real);
                const b = numberWithDecimalPoint(term.coefficient.imag);
                const c = numberWithDecimalPoint(term.power.real);
                const d = numberWithDecimalPoint(term.power.imag);
                localVarCode.push(`float _mag${i} = exp(${c}*_s - ${d}*_t);`);
                localVarCode.push(`float _phase${i} = ${c}*_t + ${d}*_s;`);
                localVarCode.push(`float _cphase${i} = cos(_phase${i});`);
                localVarCode.push(`float _sphase${i} = sin(_phase${i});`);
                xTerms.push(`${a} * _mag${i} * _cphase${i}`);
                // Extra space so if b<0 we'll get something like "- -4" which is valid.
                xTerms.push(`- ${b} * _mag${i} * _sphase${i}`);
                yTerms.push(`${b} * _mag${i} * _cphase${i}`);
                yTerms.push(`${a} * _mag${i} * _sphase${i}`);
            }
            i += 1;
        }
        return `
            ${localVarCode.join('\n')}
            float ${newXVar} = ${sumOfExpressions(xTerms)} + ${cxVar};
            float ${newYVar} = ${sumOfExpressions(yTerms)} + ${cyVar};
        `;
    }

    weightedMean(other, weight) {
        const numTerms = this.terms.length;
        const numOtherTerms = other.terms.length;
        const maxTerms = Math.max(numTerms, numOtherTerms);
        const newTerms = [];
        for (let i = 0; i < maxTerms; i++) {
            const thisTerm = (i < numTerms) ?
                this.terms[i] : new Term(Complex.ZERO, other.terms[i].power);
            const otherTerm = (i < numOtherTerms) ?
                other.terms[i] : new Term(Complex.ZERO, thisTerm.power);
            newTerms.push(thisTerm.weightedMean(otherTerm, weight));
        }
        return new Expression(newTerms);
    }

    toJson() {
        return this.terms.map(t => t.toJson());
    }

    toString() {
        return JSON.stringify(this.toJson());
    }
}

/**
 * Input is array of [coefficient, power], e.g. [[2,3], [-1,2]] for 2z^3-3z^2.
 * Output is real and imag arrays of [coefficient, x power, y power].
 * 2z^3-3z^2 = 2(x+yi)^3 - 3(x+yi)^2 = (2x^3-6xy^2-3x^2+3y^2) + i(6x^2y-2y^3-6xy) =>
 * [
 *   [[2, 3, 0], [-6, 1, 2], [-3, 2, 0], [3, 0, 2]],
 *   [[6, 2, 1], [-2, 0, 3], [-6, 1, 1]],
 * ]
 * Powers must be nonnegative integers, coefficients can be any real.
 */
const expandIntegerPolynomial = (poly) => {
    const pascalRow = (n) => {
        // 4 -> [1, 4, 6, 4, 1]
        const factorials = [1];
        for (let i=1; i<=n; i++) {
            factorials.push(factorials[i-1] * i);
        }
        const row = [];
        for (let i=0; i<=n; i++) {
            row.push(Math.round(factorials[n] / (factorials[i]*factorials[n-i])));
        }
        return row;
    };
    const realTerms = [];
    const imagTerms = [];
    for (const [coeff, power] of poly) {
        if (power < 0 || power%1 !== 0.0) {
            throw Error('Invalid power: ' + power);
        }
        const prow = pascalRow(power);
        // term t of power n is x^(n-t)*y^t. Terms alternate real and imaginary, and +/-.
        let sign = +1;
        for (let i=0; i<=power; i+=2) {
            realTerms.push([sign*coeff*prow[i], power-i, i]);
            sign = -sign;
        }
        sign = +1;
        for (let i=1; i<=power; i+=2) {
            imagTerms.push([sign*coeff*prow[i], power-i, i]);
            sign = -sign;
        }
    }
    return [realTerms, imagTerms];
};

const numberWithDecimalPoint = (a) => {
    const s = String(a);
    return (s.indexOf('.') >= 0) ? a : a+'.';
};

const sumOfExpressions = (exprs) => {
    const buffer = [];
    for (const e of exprs) {
        if (buffer.length>0 && e[0]!=='-') {
            buffer.push('+');
        }
        buffer.push(e);
    }
    return buffer.join('');
};

const integerPowersToGlExpr = (terms, xvar='x', yvar='y') => {
    const termToExpr = ([coeff, xpower, ypower]) => {
        const factors = [];
        if (coeff!==1 || (xpower===0 && ypower===0)) {
            factors.push(numberWithDecimalPoint(coeff));
        }
        for (let i=0; i<xpower; i++) {
            factors.push(xvar);
        }
        for (let i=0; i<ypower; i++) {
            factors.push(yvar);
        }
        return factors.join('*');
    };
    return sumOfExpressions(terms.map(termToExpr));
};

const vertexShaderSource = `
attribute vec4 pos;

void main() {
    gl_Position = pos;
}
`;

const generateFragmentShaderSource = (expr, colorScheme, overlayColorScheme) => {
    return `
    precision highp float;
    uniform float jx;
    uniform float jy;
    uniform float minx;
    uniform float maxx;
    uniform float miny;
    uniform float maxy;
    uniform float width;
    uniform float height;
    uniform bool showMandelbrot;
    uniform bool showJulia;
    const float maxIters = 255.0;

    float juliaIters(float cx, float cy, float jx, float jy) {
        float x = jx;
        float y = jy;
        for (float iters=0.0; iters < maxIters; iters += 1.0) {
            ${expr.toGlShaderCode()}
            x = newx;
            y = newy;
            if (x*x + y*y > 16.0) return iters;
        }
        return maxIters;
    }

    void main() {
        float xfrac = gl_FragCoord.x / width;
        float yfrac = 1.0 - (gl_FragCoord.y / height);
        float x = minx + xfrac*(maxx-minx);
        float y = maxy - yfrac*(maxy-miny);
        float mIters = showMandelbrot ? juliaIters(x, y, 0.0, 0.0) : 0.0;
        float jIters = showJulia ? juliaIters(jx, jy, x, y) : 0.0;
        float red, green, blue, iters;
        if (showMandelbrot && showJulia) {
            iters = mIters;
            float r1 = ${colorScheme.red};
            float g1 = ${colorScheme.green};
            float b1 = ${colorScheme.blue};
            iters = jIters;
            red = (2.0 * r1 + (${overlayColorScheme.red})) / 3.0;
            green = (2.0 * g1 + (${overlayColorScheme.green})) / 3.0;
            blue = (2.0 * b1 + (${overlayColorScheme.blue})) / 3.0;
        }
        else if (showMandelbrot) {
            iters = mIters;
            red = ${colorScheme.red};
            green = ${colorScheme.green};
            blue = ${colorScheme.blue};
        }
        else if (showJulia) {
            iters = jIters;
            red = ${colorScheme.red};
            green = ${colorScheme.green};
            blue = ${colorScheme.blue};
        }
        gl_FragColor = vec4(red, green, blue, 1);
    }
    `;
};

const createShader = (gl, type, source) => {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    const success = gl.getShaderParameter(shader, gl.COMPILE_STATUS);
    if (success) {
        return shader;
    }
    else {
        console.warn('createShader failed', gl.getShaderInfoLog(shader));
        gl.deleteShader(shader);
        return null;
    }
}

const createGlProgram = (gl, expr, colorScheme, overlayColorScheme) => {
    const vertexShader = createShader(gl, gl.VERTEX_SHADER, vertexShaderSource);
    const fragmentShader = createShader(
        gl, gl.FRAGMENT_SHADER, generateFragmentShaderSource(expr, colorScheme, overlayColorScheme));
    const program = gl.createProgram();
    gl.attachShader(program, vertexShader);
    gl.attachShader(program, fragmentShader);
    gl.linkProgram(program);
    return program;
};

const FractalType = {
    MANDELBROT: 'mandelbrot',
    JULIA: 'julia',
};

class ViewBounds {
    // `center` is a Complex instance.
    constructor(center, radius) {
        this.center = center;
        this.radius = radius;
    }

    pointAtFraction(frac) {
        return new Complex(this.center.real + (frac.x-0.5)*2*this.radius,
                           this.center.imag + (frac.y-0.5)*2*this.radius);
    }

    // centerFrac: center of the zoom (as ratio, so (0.8, 0.5) zooms in/out at center right).
    // zoomRatio: new x/y range is increased/decreased by this ratio (positive zooms out).
    zoomedBounds(centerFrac, zoomRatio) {
        const rangeDelta = 2 * this.radius * zoomRatio;
        const xmin = this.center.real - this.radius - rangeDelta*centerFrac.x;
        const xmax = this.center.real + this.radius + rangeDelta*(1-centerFrac.x);
        const ymin = this.center.imag - this.radius - rangeDelta*centerFrac.y;
        const ymax = this.center.imag + this.radius + rangeDelta*(1-centerFrac.y);
        return new ViewBounds(new Complex((xmin+xmax)/2, (ymin+ymax)/2), (xmax-xmin)/2);
    }
}

class FractalParams {
    constructor() {
        this.fractalType = null;
        this.bounds = new ViewBounds(Complex.ZERO, 0);
        this.expression = null;
        this.juliaSeed = null;
    }

    static fromJson(json) {
        const self = new FractalParams();
        self.fractalType = json['fractalType'];
        const center = json['center'];
        const radius = json['radius'];
        if (center && radius) {
            self.bounds = new ViewBounds(Complex.fromString(center), radius);
        }
        if (json['expression']) {
            self.expression = Expression.fromArray(json['expression']);
        }
        if (json['juliaSeed']) {
            self.juliaSeed = Complex.fromString(json['juliaSeed']);
        }
        return self;
    }

    toJson() {
        const json = {};
        if (this.fractalType) {
            json['fractalType'] = this.fractalType;
        }
        if (this.expression) {
            json['expression'] = this.expression.toJson();
        }
        if (this.bounds) {
            json['center'] = this.bounds.center.toString();
            json['radius'] = this.bounds.radius;
        }
        if (this.juliaSeed) {
            json['juliaSeed'] = this.juliaSeed.toString();
        }
        return json;
    }

    paramsForAnimationState(animation) {
        const fraction = animation.animationFraction();
        const endParams = animation.targetFractalParams;
        if (fraction === 0 || !endParams) {
            return this;
        }
        const params = new FractalParams();
        params.fractalType = this.fractalType;
        params.expression = animation.animateExpression && endParams.expression ?
            this.expression.weightedMean(endParams.expression, fraction) : this.expression;

        const isJuliaSeedAnimated = animation.animateJuliaSeed && endParams.juliaSeed &&
              this.fractalType === FractalType.JULIA;
        params.juliaSeed = isJuliaSeedAnimated ?
            this.juliaSeed.weightedMean(endParams.juliaSeed, fraction): this.juliaSeed;

        if (animation.animateBounds && endParams.bounds) {
            const startRadius = this.bounds.radius;
            const endRadius = endParams.bounds.radius;
            const radius = weightedGeometricMean(startRadius, endRadius, fraction);
            const centerRatio = (endRadius == startRadius) ?
                fraction : (radius - startRadius) / (endRadius - startRadius);
            const center = this.bounds.center.weightedMean(endParams.bounds.center, centerRatio);
            params.bounds = new ViewBounds(center, radius);
        }
        else {
            params.bounds = this.bounds;
        }
        return params;
    }
}

class Animation {
    constructor() {
        this.targetFractalParams = new FractalParams();
        this.animateBounds = false;
        this.animateExpression = false;
        this.animateJuliaSeed = false;
        this.numFrames = 500;
        this.currentFrame = 0;
        this.animationDirection = 1;
    }

    static fromJson(json) {
        const self = new Animation();
        self.targetFractalParams = FractalParams.fromJson(json['target'] || {});
        self.animateBounds = !!json['animateBounds'];
        self.animateExpression = !!json['animateExpression'];
        self.animateJuliaSeed = !!json['animateJuliaSeed'];
        self.numFrames = json['numFrames'] || 500;
        return self;
    }

    toJson() {
        const json = {};
        json['target'] = this.targetFractalParams.toJson();
        json['animateBounds'] = this.animateBounds;
        json['animateExpression'] = this.animateExpression;
        json['animateJuliaSeed'] = this.animateJuliaSeed;
        json['numFrames'] = this.numFrames;
        return json;
    }

    animationFraction() {
        return (this.numFrames > 0) ? this.currentFrame / this.numFrames : 0;
    }

    nextFrame() {
        let frame = this.currentFrame + this.animationDirection;
        if (frame > this.numFrames) {
            frame = this.numFrames;
            this.animationDirection = -1;
        }
        else if (frame < 0) {
            frame = 0;
            this.animationDirection = 1;
        }
        this.currentFrame = frame;
    }
}

const fractionalEventPosition = (event, element) => {
    const r = element.getBoundingClientRect();
    return {x: (event.clientX - r.left) / r.width, y: 1 - ((event.clientY - r.top) / r.height)};
};

const clamp = (x, min, max) => {
    return Math.min(max, Math.max(x, min));
}

const weightedMean = (x1, x2, weight) => x1 + (x2-x1)*weight;

const weightedGeometricMean = (x1, x2, weight) =>
    Math.exp(weightedMean(Math.log(x1), Math.log(x2), weight));

const roundToPlaces = (x, places) => x.toFixed(places).replace(/\.?0+$/, '');

const presets = [
    {
        name: 'Mandelbrot',
        fractalParams: {
            fractalType: FractalType.MANDELBROT,
            expression: [[1, 2]],
            center: '-0.5',
            radius: 1.5,
            juliaSeed: '-0.4 + 0.6i',
        },
    },
    {
        name: 'Mandelbrot z^4',
        fractalParams: {
            fractalType: FractalType.MANDELBROT,
            expression: [[1, 4]],
            center: '-0.25',
            radius: 1.25,
            juliaSeed: '-0.4 + 0.47i',
        },
    },
    {
        name: 'Mandelbrot powers',
        fractalParams: {
            fractalType: FractalType.MANDELBROT,
            expression: [[1, 2]],
            center: '0',
            radius: 1.5,
        },
        animationEnabled: true,
        autoplay: true,
        animation: {
            numFrames: 500,
            animateExpression: true,
            target: {
                expression: [[1, 8]],
            },
        },
    },
    {
        name: 'Seahorse valley',
        fractalParams: {
            fractalType: FractalType.MANDELBROT,
            expression: [[1, 2]],
            center: '-0.5',
            radius: 1.25,
        },
        animationEnabled: true,
        autoplay: true,
        animation: {
            numFrames: 600,
            animateBounds: true,
            target: {
                center: '-0.74635896 + 0.09853025i',
                radius: 0.00005,
            },
        },
    },
    {
        name: 'Zero to two',
        fractalParams: {
            fractalType: FractalType.MANDELBROT,
            expression: [[1, 1.998]],
            center: '-1.755',
            radius: 0.04,
        },
        animationEnabled: true,
        autoplay: true,
        animation: {
            numFrames: 500,
            animateExpression: true,
            target: {
                expression: [[1, 2.008]],
            },
        },
    },
    {
        name: 'Collisions',
        fractalParams: {
            fractalType: FractalType.MANDELBROT,
            expression: [[1, 2], [0.15, 3]],
            center: '-2.5',
            radius: 1,
        },
        animationEnabled: true,
        autoplay: true,
        animation: {
            numFrames: 1000,
            animateExpression: true,
            target: {
                expression: [[1, 2], [0.2, 3]],
            },
        },
    },
    {
        name: 'Julia dust',
        fractalParams: {
            fractalType: FractalType.JULIA,
            expression: [[1, 2]],
            center: '0',
            radius: 1.25,
            juliaSeed: '0.26',
        },
        animation: {
            animateExpression: true,
            target: {
                expression: [[1, 2.06]],
            },
        },
    },
    {
        name: 'Julia spirals',
        fractalParams: {
            fractalType: FractalType.JULIA,
            expression: [[1, 2]],
            center: '0',
            radius: 2,
            juliaSeed: '-0.75',
        },
        animationEnabled: true,
        autoplay: true,
        animation: {
            numFrames: 500,
            animateBounds: true,
            animateJuliaSeed: true,
            target: {
                center: '-0.5 + 0.1i',
                radius: 0.05,
                juliaSeed: '-0.75 + 0.2i',
                expression: [[1, 2.001]],
            },
        },
    },
];

/**
 * `red`, `green`, `blue` values are GL shader expressions which can use the following variables:
 * iters: number of iterations of current point before escaping, or `maxIters`
 * maxIters: maximum number of iterations
 * xfrac: X position of current point in the viewport, from 0 to 1.
 * yfrac: X position of current point in the viewport, from 0 to 1.
 */
const colorSchemes = [
    {
        name: 'Aquamarine',
        red: '0.0',
        green: 'iters / maxIters',
        blue: '1.0 - yfrac',
    },
    {
        name: 'Twilight',
        red: '0.9 * (iters / maxIters)',
        green: '0.0',
        blue: '0.9 * (1.0 - yfrac)',
    },
    {
        name: 'Fire',
        red: '0.75 + (0.25 * (iters + (mod(maxIters - iters, 8.0) == 0.0 ? 10.0 : 0.0)) / maxIters)',
        green: '0.13 + (0.87 * (iters + (mod(maxIters - iters, 8.0) == 0.0 ? 10.0 : 0.0)) / maxIters)',
        blue: '0.13 - (0.13 * (iters + (mod(maxIters - iters, 8.0) == 0.0 ? 10.0 : 0.0)) / maxIters)',
    },
    {
        name: 'Ice',
        red: 'mod(maxIters - iters, 8.0) == 0.0 ? 0.9 : 0.0',
        green: 'mod(maxIters - iters, 8.0) == 0.0 ? 0.9 : iters / maxIters',
        blue: 'mod(maxIters - iters, 8.0) == 0.0 ? 0.9 : iters / maxIters',
    },
    {
        name: 'Lightning',
        red: 'iters == maxIters ? 0.9 : (0.5 + 0.3 * yfrac) * (0.54 + (0.46 * iters / maxIters))',
        green: 'iters == maxIters ? 0.9 : (0.5 + 0.3 * yfrac) * (0.17 + (0.83 * iters / maxIters))',
        blue: 'iters == maxIters ? 0.9 : (0.5 + 0.3 * yfrac) * (0.89 + (0.11 * iters / maxIters))',
    },
    {
        name: 'Zebra',
        red: '1.0 - mod(iters, 2.0)',
        green: '1.0 - mod(iters, 2.0)',
        blue: '1.0 - mod(iters, 2.0)',
    },
    {
        name: 'ANSI',
        red: 'mod(maxIters - iters, 8.0) >= 4.0 ? 1.0 : 0.0',
        green: 'mod(maxIters - iters, 4.0) >= 2.0 ? 1.0 : 0.0',
        blue: 'mod(maxIters - iters, 2.0) >= 1.0 ? 1.0 : 0.0',
    },
];

Vue.component('fract-complex-number', {
    template: '<input ref="valueField" type="text" @change="updateValue($event.target.value)" />',
    props: ['value'],
    methods: {
        updateValue(formattedValue) {
            const newValue = (formattedValue) ? Complex.fromString(formattedValue) : null;
            this.$emit('input', newValue);
        },
    },
    watch: {
        value() {
            const formattedValue = (this.value) ? this.value.toRoundedString(8) : '';
            this.$refs.valueField.value = formattedValue;
        },
    },
});

Vue.component('fract-expression', {
    template: '<input ref="valueField" type="text" @change="updateValue($event.target.value)" />',
    props: ['value'],
    methods: {
        updateValue(formattedValue) {
            const newValue = (formattedValue) ? Expression.fromString(formattedValue) : null;
            this.$emit('input', newValue);
        },
    },
    watch: {
        value() {
            const formattedValue = (this.value) ? this.value.toString() : '';
            this.$refs.valueField.value = formattedValue;
        },
    },
});

const createApp = () => new Vue({
    el: '#main',

    data: {
        presets: presets,
        selectedPresetName: '',
        fractalParams: new FractalParams(),
        animation: new Animation(),

        colorSchemes: colorSchemes,
        selectedColorScheme: colorSchemes[0],

        isAnimating: false,
        animationEnabled: false,

        fractalSize: 0,
        overlayJulia: false,

        mouseDownFractionalPosition: null,

        showDebugInfo: false,
        lastDisplayedFractalParams: '',
        totalFrames: 0,

        devicePixelRatio: window.devicePixelRatio || 1,
        selectedDpiRatio: 1,
        useHighDpi: false,

        canvas: null,
        gl: null,
    },

    methods: {
        selectPreset() {
            const preset = this.presets.find(p => p.name === this.selectedPresetName);
            this.restoreStateFromJson(preset);
            const isAnimating = !!preset['autoplay'];
            if (isAnimating && !this.isAnimating) {
                this.isAnimating = true;
                window.setTimeout(() => this.animationTick(), 0);
            }
            this.redrawAndStoreState();
        },

        isHighDpiDisplay() {
            return this.devicePixelRatio > 1;
        },

        displayedFractalParams() {
            return this.animationEnabled ?
                this.fractalParams.paramsForAnimationState(this.animation) : this.fractalParams;
        },

        resetViewBounds() {
            this.fractalParams.bounds = new ViewBounds(Complex.ZERO, 2);
            this.redrawAndStoreState();
        },

        mouseMovedOverCanvas(event) {
            const updateJuliaOverlay =
                this.fractalParams.fractalType === FractalType.MANDELBROT && this.overlayJulia;
            if (updateJuliaOverlay || this.mouseDownFractionalPosition) {
                if (updateJuliaOverlay) {
                    const frac = fractionalEventPosition(event, this.canvas);
                    this.fractalParams.juliaSeed =
                        this.displayedFractalParams().bounds.pointAtFraction(frac);
                }
                if (this.mouseDownFractionalPosition) {
                    const bounds = this.fractalParams.bounds;
                    const pos = fractionalEventPosition(event, this.canvas);
                    const dx = 2 * bounds.radius * (pos.x - this.mouseDownFractionalPosition.x);
                    const dy = 2 * bounds.radius * (pos.y - this.mouseDownFractionalPosition.y);
                    this.fractalParams.bounds.center = new Complex(
                        bounds.center.real - dx, bounds.center.imag - dy);
                    this.mouseDownFractionalPosition = pos;
                }
                this.redraw();
            }
        },

        mouseWheelOverCanvas(event) {
            // TODO: Don't mess up animation.
            const frac = fractionalEventPosition(event, this.canvas);
            const delta = clamp(event.deltaY, -10, 10);
            this.fractalParams.bounds = this.fractalParams.bounds.zoomedBounds(frac, delta * 0.01);
            this.redrawAndStoreState();
        },

        mouseDownOverCanvas(event) {
            this.mouseDownFractionalPosition = fractionalEventPosition(event, this.canvas);
        },

        mouseUpOverCanvas(event) {
            this.mouseDownFractionalPosition = null;
            this.storeStateInUrl();
        },

        mouseExitedCanvas(event) {
            this.mouseDownFractionalPosition = null;
            this.storeStateInUrl();
        },

        redraw() {
            this.totalFrames += 1;

            let gl = this.gl;
            if (!gl) {
                this.gl = this.canvas.getContext('webgl');
                gl = this.gl;
                gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
                gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);
            }

            const params = this.displayedFractalParams();
            const fractalType = params.fractalType;
            const bounds = params.bounds;
            const center = bounds.center;
            const juliaSeed = params.juliaSeed || Complex.ZERO;

            const program = createGlProgram(gl, params.expression,
                                            this.selectedColorScheme, this.selectedColorScheme);
            gl.useProgram(program);
            gl.clearColor(0, 0, 0, 0);
            gl.clear(gl.COLOR_BUFFER_BIT);

            gl.uniform1f(gl.getUniformLocation(program, 'width'), this.canvas.width);
            gl.uniform1f(gl.getUniformLocation(program, 'height'), this.canvas.height);
            gl.uniform1f(gl.getUniformLocation(program, 'minx'), center.real - bounds.radius);
            gl.uniform1f(gl.getUniformLocation(program, 'maxx'), center.real + bounds.radius);
            gl.uniform1f(gl.getUniformLocation(program, 'miny'), center.imag - bounds.radius);
            gl.uniform1f(gl.getUniformLocation(program, 'maxy'), center.imag + bounds.radius);
            gl.uniform1f(gl.getUniformLocation(program, 'jx'), juliaSeed.real);
            gl.uniform1f(gl.getUniformLocation(program, 'jy'), juliaSeed.imag);
            gl.uniform1i(gl.getUniformLocation(program, 'showMandelbrot'),
                                               fractalType === FractalType.MANDELBROT ? 1 : 0);
            gl.uniform1i(gl.getUniformLocation(program, 'showJulia'),
                                               fractalType === FractalType.JULIA || this.overlayJulia ? 1 : 0);

            const positions = [-1, -1,  1, -1,  -1, 1,  1, -1,  -1, 1,  1, 1];
            const posAttr = gl.getAttribLocation(program, 'pos');
            gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(positions), gl.STATIC_DRAW);
            gl.enableVertexAttribArray(posAttr);
            gl.vertexAttribPointer(posAttr, 2 /* size */, gl.FLOAT, false /* normalize */, 0 /* stride */, 0 /* offset */);
            gl.drawArrays(gl.TRIANGLES, 0, 6);

            this.lastDisplayedFractalParams = params;
        },

        redrawAndStoreState() {
            this.redraw();
            this.storeStateInUrl();
        },

        resizeFractal() {
            this.canvas.style.width = this.fractalSize + 'px';
            this.canvas.style.height = this.fractalSize + 'px';
            this.canvas.width = this.fractalSize * this.selectedDpiRatio;
            this.canvas.height = this.fractalSize * this.selectedDpiRatio;
            this.gl = null;
            this.redraw();
        },

        animationTick() {
            if (!this.isAnimating) {
                return;
            }
            this.animation.nextFrame();
            this.redraw();
            window.setTimeout(() => this.animationTick(), 10);
        },

        toggleAnimationEnabled() {
            if (!this.animationEnabled) {
                this.isAnimating = false;
            }
            this.redraw();
        },

        toggleAnimationPlaying() {
            this.isAnimating = !this.isAnimating;
            if (this.isAnimating) {
                window.setTimeout(() => this.animationTick(), 0);
            }
        },

        storeStateInUrl() {
            if (this.mouseDownFractionalPosition) {
                return;
            }
            const json = {
                fractalParams: this.fractalParams.toJson(),
                animation: this.animation.toJson(),
                animationEnabled: this.animationEnabled,
                color: this.selectedColorScheme.name,
            };
            const url = '?' + encodeURIComponent(JSON.stringify(json));
            history.replaceState(null, '', url);
        },

        restoreStateFromUrl() {
            const query = decodeURIComponent(document.location.search);
            if (query) {
                const json = JSON.parse(query.substring(1));
                this.restoreStateFromJson(json);
            }
        },

        restoreStateFromJson(json) {
            this.fractalParams = FractalParams.fromJson(json['fractalParams'] || {});
            this.animation = Animation.fromJson(json['animation'] || {});
            this.animationEnabled = !!json['animationEnabled'];
            if (json['color']) {
                this.selectedColorScheme =
                    colorSchemes.find(s => s.name === json['color']) || colorSchemes[0];
            }
        },

        updateDpi() {
            try {
                window.localStorage['useHighDpi'] = this.useHighDpi ? '1': '';
            } catch (ignored) {}
            this.selectedDpiRatio = this.useHighDpi ? this.devicePixelRatio : 1;
            this.resizeFractal();
        },

        updateFractalSize() {
            try {
                window.localStorage['fractalSize'] = this.fractalSize;
            } catch (ignored) {}
            this.resizeFractal();
        },
    },

    mounted() {
        this.canvas = document.querySelector('#__canvas');
        let restoreError = false;
        try {
            this.restoreStateFromUrl();
        }
        catch (ex) {
            console.log('Error restoring from URL: ', ex);
            restoreError = true;
        }
        if (restoreError || !this.fractalParams.fractalType) {
            this.selectedPresetName = presets[0].name;
            this.selectPreset();
        }

        try {
            this.fractalSize = parseInt(window.localStorage['fractalSize'], 10);
            this.useHighDpi = !!window.localStorage['useHighDpi'];
            this.selectedDpiRatio = this.useHighDpi ? this.devicePixelRatio : 1;
        } catch (ignored) {}
        if (!(Number.isFinite(this.fractalSize) && this.fractalSize > 0)) {
            this.fractalSize = 800;
        }

        this.resizeFractal();
    },
});

document.addEventListener('DOMContentLoaded', createApp);
})();
