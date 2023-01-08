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

    useIntegerPowerFastPath() {
        return this.terms.every(t =>
            t.coefficient.isRealOnly() && t.power.isRealNonnegativeInteger() && t.power <= 8);
    }

    canUse64BitFloats() {
        return this.terms.every(
            t => t.coefficient.isRealOnly() && t.power.isRealNonnegativeInteger());
    }

    toGlShaderCode(xVar='x', yVar='y', newXVar='newx', newYVar='newy', cxVar='cx', cyVar='cy') {
        // For integer powers we only need multiplications. Except that's slower than the general
        // approach for large powers because of the number of factors. Caching intermediate powers
        // might help (e.g. 'float x2=x*x; float x3=x2*x; ...').
        if (this.useIntegerPowerFastPath()) {
            const integerPoly = this.terms.map(t => [t.coefficient.real, t.power.real]);
            const [realParts, imagParts] = expandIntegerPolynomial(integerPoly);
            return `
                float ${newXVar} = ${integerPowersToGlExpr(realParts)} + ${cxVar};
                float ${newYVar} = ${integerPowersToGlExpr(imagParts)} + ${cyVar};
            `;
        }
        // ln(x+yi) = s + ti. atan(0, 0) causes failures in some browsers so have to check for it.
        const localVarCode = [
            'float _s = 0.5*log(x*x + y*y);',
            'float _t = (x == 0.0 && y == 0.0) ? 0.0 : atan(y, x);'
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

    toF64GlShaderCode(xVar='x', yVar='y', newXVar='newx', newYVar='newy', cxVar='cx', cyVar='cy') {
        const maxPower = Math.max.apply(null, this.terms.map(t => t.power.real));
        const integerPoly = this.terms.map(t => [t.coefficient.real, t.power.real]);
        const [realParts, imagParts] = expandIntegerPolynomial(integerPoly);
        // Create temp vars for all x and y powers.
        const codeLines = [];
        if (maxPower >= 2) {
            codeLines.push(`vec2 _x2 = f64_mul(${xVar}, ${xVar});`);
            codeLines.push(`vec2 _y2 = f64_mul(${yVar}, ${yVar});`);
            for (let p=3; p<=maxPower; p++) {
                codeLines.push(`vec2 _x${p} = f64_mul(_x${p-1}, x);`);
                codeLines.push(`vec2 _y${p} = f64_mul(_y${p-1}, y);`);
            }
        }

        const makeTerm = ([coeff, xPower, yPower]) => {
            if (coeff === 0) {
                // 0*anything = 0
                return `vec2(0.)`;
            }
            if (xPower === 0 && yPower === 0) {
                // a * x^0 * y^0 = a
                return `vec2(${numberWithDecimalPoint(coeff)})`;
            }
            const coeffPrefix = (coeff === 1) ? '' : `${numberWithDecimalPoint(coeff)} * `;
            const xPowerVar = (xPower <= 1) ? xVar : `_x${xPower}`;
            const yPowerVar = (yPower <= 1) ? yVar : `_y${yPower}`;
            if (xPower === 0) {
                // a * y^n
                return `${coeffPrefix}${yPowerVar}`;
            }
            else if (yPower === 0) {
                // a * x^n
                return `${coeffPrefix}${xPowerVar}`;
            }
            else {
                // a * x^n * y^n
                return `${coeffPrefix}f64_mul(${xPowerVar}, ${yPowerVar})`;
            }
        }

        codeLines.push(`vec2 ${newXVar} = ${cxVar};`);
        for (const part of realParts) {
            codeLines.push(`${newXVar} = f64_add(${newXVar}, ${makeTerm(part)});`);
        }
        codeLines.push(`vec2 ${newYVar} = ${cyVar};`);
        for (const part of imagParts) {
            codeLines.push(`${newYVar} = f64_add(${newYVar}, ${makeTerm(part)});`);
        }

        return codeLines.join('\n');
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

const generateFragmentShaderSource = (expr, colorScheme, overlayColorScheme, maxIters) => {
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
    const float maxIters =  ${numberWithDecimalPoint(maxIters)};

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

// https://www.thasler.com/blog/blog/glsl-part2-emu
const generate64BitFragmentSource = (expr, colorScheme, overlayColorScheme, maxIters) => {
    return `
    precision highp float;
    uniform float jxHigh;
    uniform float jxLow;
    uniform float jyHigh;
    uniform float jyLow;
    uniform float minxHigh;
    uniform float minxLow;
    uniform float maxxHigh;
    uniform float maxxLow;
    uniform float minyHigh;
    uniform float minyLow;
    uniform float maxyHigh;
    uniform float maxyLow;
    uniform float width;
    uniform float height;
    uniform bool showMandelbrot;
    uniform bool showJulia;
    const float maxIters = ${numberWithDecimalPoint(maxIters)};

    vec2 f64_add(vec2 f1, vec2 f2) {
        float t1 = f1.x + f2.x;
        float e = t1 - f1.x;
        float t2 = ((f2.x - e) + (f1.x - (t1 - e))) + f1.y + f2.y;
        float high = t1 + t2;
        return vec2(high, t2 - (high - t1));
    }

    vec2 f64_mul(vec2 f1, vec2 f2) {
        float cona = f1.x * 8193.;
        float conb = f2.x * 8193.;
        float a1 = cona - (cona - f1.x);
        float b1 = conb - (conb - f2.x);
        float a2 = f1.x - a1;
        float b2 = f2.x - b1;
        float c11 = f1.x * f2.x;
        float c21 = a2 * b2 + (a2 * b1 + (a1 * b2 + (a1 * b1 - c11)));
        float c2 = f1.x * f2.y + f1.y * f2.x;
        float t1 = c11 + c2;
        float e = t1 - c11;
        float t2 = f1.y * f2.y + ((c2 - e) + (c11 - (t1 - e))) + c21;
        float high = t1 + t2;
        return vec2(high, t2 - (high - t1));
    }

    float juliaIters(vec2 cx, vec2 cy, vec2 jx, vec2 jy) {
        vec2 x = jx;
        vec2 y = jy;
        for (float iters=0.0; iters < maxIters; iters += 1.0) {
            ${expr.toF64GlShaderCode()}
            x = newx;
            y = newy;
            if (x.x*x.x + y.x*y.x > 16.0) return iters;
        }
        return maxIters;
    }

    void main() {
        float xfrac = gl_FragCoord.x / width;
        float yfrac = 1.0 - (gl_FragCoord.y / height);
        vec2 deltaX = f64_add(vec2(maxxHigh, maxxLow), vec2(-minxHigh, -minxLow));
        vec2 deltaY = f64_add(vec2(maxyHigh, maxyLow), vec2(-minyHigh, -minyLow));

        vec2 x = f64_add(vec2(minxHigh, minxLow), xfrac * deltaX);
        vec2 y = f64_add(vec2(maxyHigh, maxyLow), -yfrac * deltaY);

        float mIters = showMandelbrot ? juliaIters(x, y, vec2(0.0), vec2(0.0)) : 0.0;
        float jIters = showJulia ? juliaIters(vec2(jxHigh, jxLow), vec2(jyHigh, jyLow), x, y) : 0.0;
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

const splitDoubleToFloats = (x) => {
    const f = new Float32Array(2);
    f[0] = x;
    f[1] = x - f[0];
    return f;
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

const createGlProgram = (gl, expr, colorScheme, overlayColorScheme, maxIters=255, use64Bit=false) => {
    const vertexShader = createShader(gl, gl.VERTEX_SHADER, vertexShaderSource);
    const fragmentShader = createShader(gl, gl.FRAGMENT_SHADER,
        use64Bit ? generate64BitFragmentSource(expr, colorScheme, overlayColorScheme, maxIters)
                 : generateFragmentShaderSource(expr, colorScheme, overlayColorScheme, maxIters));
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
        return new Complex(this.center.real + frac.x * this.radius,
                           this.center.imag + frac.y * this.radius);
    }
}

class FractalParams {
    constructor() {
        this.fractalType = null;
        this.overlayJulia = false;
        this.bounds = new ViewBounds(Complex.ZERO, 0);
        this.expression = null;
        this.juliaSeed = null;
        this.maxIters = 255;
        this.use64Bit = false;
    }

    copy() {
        const p = new FractalParams();
        p.fractalType = this.fractalType;
        p.overlayJulia = this.overlayJulia;
        p.bounds = new ViewBounds();
        p.bounds.center = this.bounds.center;
        p.bounds.radius = this.bounds.radius;
        p.expression = this.expression;
        p.juliaSeed = this.juliaSeed;
        p.maxIters = this.maxIters;
        p.use64Bit = this.use64Bit;
        return p;
    }

    static fromJson(json) {
        const self = new FractalParams();
        self.fractalType = json['fractalType'];
        self.overlayJulia = !!json['overlayJulia'];
        self.maxIters = json['maxIters'] || 255;
        self.use64Bit = !!json['use64Bit'];
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
        if (this.overlayJulia) {
            json['overlayJulia'] = 1;
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
        if (this.maxIters) {
            json['maxIters'] = this.maxIters;
        }
        if (this.use64Bit) {
            json['use64Bit'] = 1;
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
        params.overlayJulia = this.overlayJulia;
        params.maxIters = this.maxIters;
        params.use64Bit = this.use64Bit;
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

class Snapshot {
    constructor() {
        this.params = null;
        this.colorScheme = null;
        this.id = null;
    }
}

const fractionalEventPosition = (event, element) => {
    const r = element.getBoundingClientRect();
    const radius = Math.min(r.width, r.height) / 2;
    const cx = r.left + r.width / 2;
    const cy = r.top + r.height / 2;
    return {x: (event.clientX - cx) / radius, y: -(event.clientY - cy) / radius};
};

const clamp = (x, min, max) => {
    return Math.min(max, Math.max(x, min));
}

const weightedMean = (x1, x2, weight) => x1 + (x2-x1)*weight;

const weightedGeometricMean = (x1, x2, weight) =>
    Math.exp(weightedMean(Math.log(x1), Math.log(x2), weight));

const roundToPlaces = (x, places) => x.toFixed(places).replace(/\.?0+$/, '');

const drawFractal = (gl, fractalParams, colorScheme, altColorScheme) => {
    const fractalType = fractalParams.fractalType;
    const bounds = fractalParams.bounds;
    const center = bounds.center;
    const juliaSeed = fractalParams.juliaSeed || Complex.ZERO;

    const pixelIncr = 2 * bounds.radius / Math.min(gl.canvas.width, gl.canvas.height);
    const use64Bit = (fractalParams.expression.canUse64BitFloats() && pixelIncr < 5e-8);
    console.log(use64Bit, pixelIncr);

    const program = createGlProgram(gl, fractalParams.expression, colorScheme, altColorScheme,
                                    fractalParams.maxIters, use64Bit);
    gl.useProgram(program);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);

    const aspectRatio = gl.canvas.width / gl.canvas.height;
    const xFactor = Math.max(aspectRatio, 1);
    const yFactor = Math.max(1 / aspectRatio, 1);

    gl.uniform1f(gl.getUniformLocation(program, 'width'), gl.canvas.width);
    gl.uniform1f(gl.getUniformLocation(program, 'height'), gl.canvas.height);
    gl.uniform1i(gl.getUniformLocation(program, 'showMandelbrot'),
                 fractalType === FractalType.MANDELBROT ? 1 : 0);
    gl.uniform1i(gl.getUniformLocation(program, 'showJulia'),
                 fractalType === FractalType.JULIA || fractalParams.overlayJulia ? 1 : 0);
    if (use64Bit) {
        const set64BitValues = (highVar, lowVar, val) => {
            const f = splitDoubleToFloats(val);
            gl.uniform1f(gl.getUniformLocation(program, highVar), f[0]);
            gl.uniform1f(gl.getUniformLocation(program, lowVar), f[1]);
        }
        set64BitValues('minxHigh', 'minxLow', center.real - xFactor * bounds.radius);
        set64BitValues('maxxHigh', 'maxxLow', center.real + xFactor * bounds.radius);
        set64BitValues('minyHigh', 'minyLow', center.imag - yFactor * bounds.radius);
        set64BitValues('maxyHigh', 'maxyLow', center.imag + yFactor * bounds.radius);
        set64BitValues('jxHigh', 'jxLow', juliaSeed.real);
        set64BitValues('jyHigh', 'jyLow', juliaSeed.imag);
    }
    else {
        gl.uniform1f(gl.getUniformLocation(program, 'minx'), center.real - xFactor * bounds.radius);
        gl.uniform1f(gl.getUniformLocation(program, 'maxx'), center.real + xFactor * bounds.radius);
        gl.uniform1f(gl.getUniformLocation(program, 'miny'), center.imag - yFactor * bounds.radius);
        gl.uniform1f(gl.getUniformLocation(program, 'maxy'), center.imag + yFactor * bounds.radius);
        gl.uniform1f(gl.getUniformLocation(program, 'jx'), juliaSeed.real);
        gl.uniform1f(gl.getUniformLocation(program, 'jy'), juliaSeed.imag);
    }

    const positions = [-1, -1,  1, -1,  -1, 1,  1, -1,  -1, 1,  1, 1];
    const posAttr = gl.getAttribLocation(program, 'pos');
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(positions), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(posAttr);
    gl.vertexAttribPointer(posAttr, 2 /* size */, gl.FLOAT, false /* normalize */, 0 /* stride */, 0 /* offset */);
    gl.drawArrays(gl.TRIANGLES, 0, 6);
};

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
            radius: 1.5,
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
        name: '1e12 zoom',
        fractalParams: {
            fractalType: FractalType.MANDELBROT,
            expression: [[1, 2]],
            center: '-0.5',
            radius: 1.5,
            maxIters: 500,
            use64Bit: true,
        },
        animationEnabled: true,
        autoplay: true,
        animation: {
            numFrames: 1000,
            animateBounds: true,
            target: {
                center: '-0.8577248987863955 + 0.24031942994105096i',
                radius: 1.5e-12,
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
    {
        name: 'Julia z^4 twist',
        fractalParams: {
            fractalType: FractalType.JULIA,
            expression: [[1, 4]],
            center: '0',
            radius: 1.3,
            juliaSeed: '-0.45 + 0.47i',
        },
        animationEnabled: true,
        autoplay: true,
        animation: {
            numFrames: 500,
            animateJuliaSeed: true,
            target: {
                juliaSeed: '-0.39 + 0.47i',
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
        red: '0.75 + ((0.25 * iters / maxIters) + (mod(maxIters - iters, 8.0) == 0.0 ? 0.05 : 0.))',
        green: '0.13 + ((0.87 * iters / maxIters) + (mod(maxIters - iters, 8.0) == 0.0 ? 0.05 : 0.))',
        blue: '0.13 - ((0.13 * iters / maxIters) + (mod(maxIters - iters, 8.0) == 0.0 ? 0.05 : 0.))',
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
            const formattedValue = (this.value) ? this.value.toRoundedString(16) : '';
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

Vue.component('fract-snapshot', {
    template: `
      <div class="snapshot control-row">
        <div class="snapshot-image">
          <canvas ref="imageCanvas" width="180" height="135"></canvas>
        </div>
        <div class="snapshot-controls">
          <button @click="$emit('restore', snapshot)">Restore</button>
          <button @click="$emit('remove', snapshot)">Remove</button>
        </div>
      </div>
    `,
    props: ['snapshot'],
    methods: {
        restore() {
            this.$emit();
        },
    },
    mounted() {
        const canvas = this.$refs.imageCanvas;
        // Always use high DPI for snapshot canvas.
        const pixelRatio = window.devicePixelRatio || 1;
        canvas.style.width = canvas.width + 'px';
        canvas.style.height = canvas.height + 'px';
        canvas.width = canvas.width * pixelRatio;
        canvas.height = canvas.height * pixelRatio;
        const gl = canvas.getContext('webgl');
        gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
        gl.viewport(0, 0, canvas.width, canvas.height);
        drawFractal(gl, this.snapshot.params, this.snapshot.colorScheme, this.snapshot.colorScheme);
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
        isFullScreen: false,

        snapshots: [],
        nextSnapshotId: 1,

        mouseDownFractionalPosition: null,

        showDebugInfo: false,
        totalFrames: 0,
        lastDisplayedFractalParams: null,

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

        xRadiusFactor() {
            const aspectRatio = this.canvas.width / this.canvas.height;
            return Math.max(aspectRatio, 1);
        },

        yRadiusFactor() {
            const aspectRatio = this.canvas.width / this.canvas.height;
            return Math.max(1 / aspectRatio, 1);
        },

        mouseMovedOverCanvas(event) {
            const params = this.fractalParams;
            const updateJuliaOverlay =
                params.fractalType === FractalType.MANDELBROT && params.overlayJulia;
            if (updateJuliaOverlay || this.mouseDownFractionalPosition) {
                if (updateJuliaOverlay) {
                    const frac = fractionalEventPosition(event, this.canvas);
                    params.juliaSeed = this.displayedFractalParams().bounds.pointAtFraction(frac);
                }
                if (this.mouseDownFractionalPosition) {
                    const bounds = params.bounds;
                    const pos = fractionalEventPosition(event, this.canvas);
                    const dx = bounds.radius * (pos.x - this.mouseDownFractionalPosition.x);
                    const dy = bounds.radius * (pos.y - this.mouseDownFractionalPosition.y);
                    this.fractalParams.bounds.center = new Complex(
                        bounds.center.real - dx, bounds.center.imag - dy);
                    this.mouseDownFractionalPosition = pos;
                }
                this.redraw();
            }
        },

        mouseWheelOverCanvas(event) {
            const frac = fractionalEventPosition(event, this.canvas);
            const xrf = this.xRadiusFactor();
            const yrf = this.yRadiusFactor();
            // Convert fract from [-xrf, xrf] to [0, 1]. (Same for y).
            const xFrac = (frac.x + xrf) / (2 * xrf);
            const yFrac = (frac.y + yrf) / (2 * yrf);

            // Positive zoomFraction zooms out, negative in.
            const zoomFraction = clamp(event.deltaY, -10, 10) * 0.01;
            const bounds = this.fractalParams.bounds;

            const radiusDelta = zoomFraction * bounds.radius;
            const newRadius = radiusDelta + bounds.radius;

            const cx = bounds.center.real;
            const cy = bounds.center.imag;
            // If we're zooming in/out from the right side, most of the change is on the left.
            const xmin = cx - xrf*bounds.radius - 2*xrf*radiusDelta*xFrac;
            const xmax = cx + xrf*bounds.radius + 2*xrf*radiusDelta*(1-xFrac);
            const ymin = cy - yrf*bounds.radius - 2*yrf*radiusDelta*yFrac;
            const ymax = cy + yrf*bounds.radius + 2*yrf*radiusDelta*(1-yFrac);

            this.fractalParams.bounds =
                new ViewBounds(new Complex((xmin+xmax)/2, (ymin+ymax)/2), newRadius)
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

            if (!this.gl) {
                this.gl = this.canvas.getContext('webgl');
                this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.gl.createBuffer());
                this.gl.viewport(0, 0, this.gl.canvas.width, this.gl.canvas.height);
            }

            const params = this.displayedFractalParams();
            drawFractal(this.gl, params, this.selectedColorScheme, this.selectedColorScheme);
            this.lastDisplayedFractalParams = params;
        },

        redrawAndStoreState() {
            this.redraw();
            this.storeStateInUrl();
        },

        resizeFractal() {
            if (this.isFullScreen) {
                const bounds = this.$el.getBoundingClientRect();
                this.canvas.style.width = window.screen.width + 'px';
                this.canvas.style.height = window.screen.height + 'px';
                this.canvas.width = window.screen.width * this.selectedDpiRatio;
                this.canvas.height = window.screen.height * this.selectedDpiRatio;
            }
            else {
                this.canvas.style.width = this.fractalSize + 'px';
                this.canvas.style.height = this.fractalSize + 'px';
                this.canvas.width = this.fractalSize * this.selectedDpiRatio;
                this.canvas.height = this.fractalSize * this.selectedDpiRatio;
            }
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

        toggleFullScreen() {
            if (this.isFullScreen) {
                (document.exitFullscreen && document.exitFullscreen()) ||
                (document.webkitExitFullscreen && document.webkitExitFullscreen()) ||
                (document.mozCancelFullScreen && document.mozCancelFullScreen()) ||
                (document.msExitFullscreen && document.msExitFullscreen());
                this.isFullScreen = false;
            }
            else {
                (this.$el.requestFullScreen ||
                 this.$el.webkitRequestFullScreen ||
                 this.$el.mozRequestFullScreen ||
                 this.$el.msRequestFullScreen).bind(this.$el)();
            }
        },

        saveParamsSnapshot() {
            const snapshot = new Snapshot();
            snapshot.params = this.displayedFractalParams().copy();
            snapshot.colorScheme = this.selectedColorScheme;
            snapshot.id = this.nextSnapshotId;
            this.nextSnapshotId += 1;
            this.snapshots.push(snapshot);
        },

        restoreSnapshot(snapshot) {
            this.fractalParams = snapshot.params.copy();
            this.selectedColorScheme = snapshot.colorScheme;
            this.redraw();
            this.storeStateInUrl();
        },

        removeSnapshot(snapshot) {
            const index = this.snapshots.indexOf(snapshot);
            if (index >= 0) {
                this.snapshots.splice(index, 1);
            }
        },
    },

    mounted() {
        this.canvas = this.$refs['mainCanvas'];
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

        const fullScreenEvents = ['fullscreenchange', 'webkitfullscreenchange',
                                  'mozfullscreenchange', 'msfullscreenchange'];
        for (const event of fullScreenEvents) {
            document.addEventListener(event, () => {
                this.isFullScreen =
                    !!(document.fullScreenElement ||
                       document.webkitFullscreenElement ||
                       document.mozFullScreenElement ||
                       document.msFullscreenElement);
                console.log('Got fullscreen event:', this.isFullScreen);
                window.setTimeout(() => this.resizeFractal());
            });
        }

        this.resizeFractal();
    },
});

document.addEventListener('DOMContentLoaded', createApp);
})();
