// Ordered compensated CSR arithmetic, owned by Numerics.
// Mirrors compensated.py EFT operations in the selected binary32/binary64
// dtype. Strict compiler flags prohibit reassociation/contraction. Buffers are
// borrowed only during this synchronous call; no Python/JAX call or retained
// state enters the GIL-free kernel. SuperLU ownership remains with its caller.
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <initializer_list>

namespace {
struct Buffers {
    Py_buffer views[10]{};
    int acquired = 0;
    ~Buffers() { while (acquired) PyBuffer_Release(&views[--acquired]); }
    bool get(PyObject* object, bool output) {
        int flags = PyBUF_FORMAT | PyBUF_C_CONTIGUOUS;
        if (output) flags |= PyBUF_WRITABLE;
        if (PyObject_GetBuffer(object, &views[acquired], flags) < 0) return false;
        ++acquired;
        return true;
    }
};
bool fail(const char* message) { PyErr_SetString(PyExc_TypeError, message); return false; }
bool same_shape(const Py_buffer& a, const Py_buffer& b) {
    if (a.ndim != b.ndim) return false;
    for (int i = 0; i != a.ndim; ++i) if (a.shape[i] != b.shape[i]) return false;
    return true;
}
const char* format(const Py_buffer& a) {
    const char* f = a.format ? a.format : "";
    return (*f == '@' || *f == '=') ? f + 1 : f;
}
int numeric_kind(const Py_buffer& a) {
    const char* f = format(a);
    if (!std::strcmp(f, "f") && a.itemsize == 4) return 0;
    if (!std::strcmp(f, "d") && a.itemsize == 8) return 1;
    if (!std::strcmp(f, "Zf") && a.itemsize == 8) return 2;
    if (!std::strcmp(f, "Zd") && a.itemsize == 16) return 3;
    return -1;
}
bool index_buffer(const Py_buffer& a) {
    const char* f = format(a);
    return a.ndim == 1 && (a.itemsize == 4 || a.itemsize == 8) &&
        (!std::strcmp(f, "i") || !std::strcmp(f, "l") || !std::strcmp(f, "q"));
}
std::int64_t index_at(const Py_buffer& a, Py_ssize_t i) {
    const char* p = static_cast<const char*>(a.buf) + i * a.itemsize;
    if (a.itemsize == 4) { std::int32_t x; std::memcpy(&x, p, 4); return x; }
    std::int64_t x; std::memcpy(&x, p, 8); return x;
}
bool csr(const Py_buffer& data, const Py_buffer& indices, const Py_buffer& ptr,
         Py_ssize_t rows, Py_ssize_t columns) {
    if (data.ndim != 1 || !index_buffer(indices) || !index_buffer(ptr) ||
        indices.shape[0] != data.shape[0] || ptr.shape[0] != rows + 1)
        return fail("CSR buffer dimensions disagree");
    if (index_at(ptr, 0) != 0 || index_at(ptr, rows) != data.shape[0])
        return fail("CSR offsets do not cover their data buffer");
    std::int64_t previous = 0;
    for (Py_ssize_t r = 1; r <= rows; ++r) {
        auto next = index_at(ptr, r);
        if (next < previous || next > data.shape[0]) return fail("CSR offset outside data buffer");
        previous = next;
    }
    for (Py_ssize_t i = 0; i < data.shape[0]; ++i) {
        auto column = index_at(indices, i);
        if (column < 0 || column >= columns) return fail("CSR column outside RHS buffer");
    }
    return true;
}
// Explicit stores enforce selected-width rounding independently of register
// width; build flags additionally forbid contraction across these boundaries.
template<class T> inline T rounded(T x) { volatile T y = x; return y; }
template<class T> struct Pair { T hi, lo; };
template<class T> Pair<T> two_sum(T a, T b) {
    T s = rounded<T>(a + b), bv = rounded<T>(s - a), av = rounded<T>(s - bv);
    T br = rounded<T>(b - bv), ar = rounded<T>(a - av);
    return {s, rounded<T>(ar + br)};
}
template<class T> Pair<T> add(Pair<T> a, Pair<T> b) {
    auto high = two_sum(a.hi, b.hi), low = two_sum(a.lo, b.lo);
    auto middle = two_sum(high.lo, low.hi), merged = two_sum(high.hi, middle.hi);
    T tail = rounded<T>(rounded<T>(low.lo + middle.lo) + merged.lo);
    return two_sum(merged.hi, tail);
}
template<class T> Pair<T> split(T x) {
    int exponent = 0;
    T m = std::frexp(x, &exponent);
    constexpr int fraction_bits = std::numeric_limits<T>::digits - 1;
    T splitter = static_cast<T>((std::uint64_t(1) << ((fraction_bits + 2) / 2)) + 1);
    T scaled = rounded<T>(splitter * m);
    T high = rounded<T>(scaled - rounded<T>(scaled - m));
    return {high, rounded<T>(m - high)};
}
template<class T> Pair<T> two_product(T a, T b) {
    int ae = 0, be = 0;
    T am = std::frexp(a, &ae), bm = std::frexp(b, &be);
    auto as = split(am), bs = split(bm);
    T p = rounded<T>(am * bm), hp = rounded<T>(as.hi * bs.hi);
    T e1 = rounded<T>(p - hp), lh = rounded<T>(as.lo * bs.hi);
    T e2 = rounded<T>(e1 - lh), hl = rounded<T>(as.hi * bs.lo);
    T e3 = rounded<T>(e2 - hl), ll = rounded<T>(as.lo * bs.lo);
    T e = rounded<T>(ll - e3);
    return {rounded<T>(std::ldexp(p, ae + be)), rounded<T>(std::ldexp(e, ae + be))};
}
template<class T> Pair<T> multiply(Pair<T> a, Pair<T> b) {
    return add(add(two_product(a.hi, b.hi), two_product(a.hi, b.lo)),
               add(two_product(a.lo, b.hi), two_product(a.lo, b.lo)));
}
template<class T> Pair<T> negate(Pair<T> a) { return {-a.hi, -a.lo}; }
template<class T> struct Value { Pair<T> real, imag; };
template<class T> Value<T> pack(Pair<T> r, Pair<T> i) {
    // Match compensated._pack_complex's explicit 1j multiply and addition,
    // including zero/nonfinite arithmetic rather than silently dropping it.
    T zr = rounded<T>(rounded<T>(T(0) * i.hi) - rounded<T>(T(1) * T(0)));
    T zi = rounded<T>(rounded<T>(T(0) * T(0)) + rounded<T>(T(1) * i.hi));
    T lr = rounded<T>(rounded<T>(T(0) * i.lo) - rounded<T>(T(1) * T(0)));
    T li = rounded<T>(rounded<T>(T(0) * T(0)) + rounded<T>(T(1) * i.lo));
    return {{rounded<T>(r.hi + zr), rounded<T>(r.lo + lr)},
            {rounded<T>(T(0) + zi), rounded<T>(T(0) + li)}};
}
template<class T, bool Complex> Value<T> sum(Value<T> a, Value<T> b) {
    auto r = add(a.real, b.real);
    if constexpr (Complex) return pack(r, add(a.imag, b.imag));
    return {r, {0, 0}};
}
template<class T, bool Complex> Value<T> product(Value<T> a, Value<T> b) {
    auto r = multiply(a.real, b.real);
    if constexpr (Complex) {
        r = add(r, negate(multiply(a.imag, b.imag)));
        auto i = add(multiply(a.real, b.imag), multiply(a.imag, b.real));
        return pack(r, i);
    }
    return {r, {0, 0}};
}
template<class T> T load(const Py_buffer& b, Py_ssize_t element, int component = 0) {
    T value;
    std::memcpy(&value, static_cast<const char*>(b.buf) + element * b.itemsize + component * sizeof(T), sizeof(T));
    return value;
}
template<class T> void store(const Py_buffer& b, Py_ssize_t element, T value, int component = 0) {
    std::memcpy(static_cast<char*>(b.buf) + element * b.itemsize + component * sizeof(T), &value, sizeof(T));
}
template<class T, bool Complex> Value<T> read_pair(const Py_buffer& hi, const Py_buffer& lo, Py_ssize_t element) {
    return {{load<T>(hi, element), load<T>(lo, element)},
        {Complex ? load<T>(hi, element, 1) : T(0), Complex ? load<T>(lo, element, 1) : T(0)}};
}
template<class T, bool Complex> Value<T> component_product(const Py_buffer& data,
        const Py_buffer& indices, const Py_buffer& ptr, const Py_buffer& xh,
        const Py_buffer& xl, Py_ssize_t row, Py_ssize_t rhs, Py_ssize_t width) {
    Value<T> value{{0, 0}, {0, 0}};
    // Each independent row consumes stored CSR entries in exactly their order.
    for (auto k = index_at(ptr, row); k < index_at(ptr, row + 1); ++k) {
        auto column = index_at(indices, k);
        Value<T> coefficient{{load<T>(data, k), 0}, {Complex ? load<T>(data, k, 1) : T(0), 0}};
        value = sum<T, Complex>(value, product<T, Complex>(coefficient,
                    read_pair<T, Complex>(xh, xl, column * width + rhs)));
    }
    return value;
}
template<class T, bool Complex> void kernel(const Py_buffer* b, Py_ssize_t rows, Py_ssize_t width) {
    // Same high traversal, low traversal, then pair combination authority.
    for (Py_ssize_t row = 0; row < rows; ++row) for (Py_ssize_t rhs = 0; rhs < width; ++rhs) {
        auto hi = component_product<T, Complex>(b[0], b[1], b[2], b[6], b[7], row, rhs, width);
        auto lo = component_product<T, Complex>(b[3], b[4], b[5], b[6], b[7], row, rhs, width);
        auto value = sum<T, Complex>(hi, lo);
        auto offset = row * width + rhs;
        store<T>(b[8], offset, value.real.hi); store<T>(b[9], offset, value.real.lo);
        if constexpr (Complex) { store<T>(b[8], offset, value.imag.hi, 1); store<T>(b[9], offset, value.imag.lo, 1); }
    }
}
PyObject* csr_pair_product(PyObject*, PyObject* args) {
    PyObject* objects[10];
    if (!PyArg_UnpackTuple(args, "csr_pair_product", 10, 10, &objects[0], &objects[1],
         &objects[2], &objects[3], &objects[4], &objects[5], &objects[6], &objects[7], &objects[8], &objects[9])) return nullptr;
    Buffers b;
    for (int i = 0; i != 10; ++i) if (!b.get(objects[i], i >= 8)) return nullptr;
    auto& x = b.views[6]; auto& output = b.views[8];
    int kind = numeric_kind(x);
    if (kind < 0) { fail("selected dtype must be native float32/64 or complex64/128"); return nullptr; }
    for (int i : {0, 3, 7, 8, 9}) if (numeric_kind(b.views[i]) != kind) {
        fail("coefficient, RHS and output selected dtypes must agree"); return nullptr;
    }
    if ((x.ndim != 1 && x.ndim != 2) || output.ndim != x.ndim ||
        !same_shape(x, b.views[7]) || !same_shape(output, b.views[9]) ||
        (x.ndim == 2 && output.shape[1] != x.shape[1])) {
        fail("RHS and output dimensions disagree"); return nullptr;
    }
    Py_ssize_t rows = output.shape[0], width = x.ndim == 1 ? 1 : x.shape[1];
    if (!csr(b.views[0], b.views[1], b.views[2], rows, x.shape[0]) ||
        !csr(b.views[3], b.views[4], b.views[5], rows, x.shape[0])) return nullptr;
    for (int i = 8; i != 10; ++i) for (int j = 0; j < i; ++j) {
        auto begin = reinterpret_cast<std::uintptr_t>(b.views[i].buf);
        auto other = reinterpret_cast<std::uintptr_t>(b.views[j].buf);
        if (b.views[i].len && b.views[j].len && begin < other + b.views[j].len && other < begin + b.views[i].len) {
            fail("output buffers must not alias input or output buffers"); return nullptr;
        }
    }
    Py_BEGIN_ALLOW_THREADS
    switch (kind) {
        case 0: kernel<float, false>(b.views, rows, width); break;
        case 1: kernel<double, false>(b.views, rows, width); break;
        case 2: kernel<float, true>(b.views, rows, width); break;
        case 3: kernel<double, true>(b.views, rows, width); break;
    }
    Py_END_ALLOW_THREADS
    if (PyErr_CheckSignals() < 0) return nullptr;
    Py_RETURN_NONE;
}
PyMethodDef methods[] = {{"csr_pair_product", csr_pair_product, METH_VARARGS,
    "Ordered same-dtype compensated CSR product into borrowed output buffers."}, {nullptr, nullptr, 0, nullptr}};
PyModuleDef module = {PyModuleDef_HEAD_INIT, "_compensated_cpu", "Call-local ordered compensated kernels.", -1, methods};
} // namespace
PyMODINIT_FUNC PyInit__compensated_cpu() { return PyModule_Create(&module); }
