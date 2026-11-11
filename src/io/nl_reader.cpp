// AMPL NL reader (text and binary formats).
//
// Written from the public specifications, no vendored source; opcode numbers
// follow AMPL's `opcode.hd`:
//   * David M. Gay, "Writing .nl Files" — https://ampl.github.io/nlwrite.pdf
//   * David M. Gay, "Hooking Your Solver to AMPL" — https://ampl.com/REFS/hooking2.pdf
//
// Text ('g') and binary ('b', host byte order only). Records the raw expression
// graph and linear parts; mapping to CBLS ops, and which opcodes are
// supported, lives in nl_to_model.cpp.

#include "cbls/io_nl.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <initializer_list>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace cbls {

namespace {

// A forward cursor over an NL file, in either encoding. The header is text in
// both ('g' and 'b' files alike), read with `peek_char`/`read_line`. After it,
// the segments carry the same items in the same order in both encodings; only
// their spelling differs:
//
//   item                      text ('g')                binary ('b')
//   segment / expression tag  one letter                one byte
//   number attached to a tag  digits glued to it: "C3"  int32 / float64
//   later numbers             whitespace-separated      int32 / float64
//   bound-record type         a digit                   one ASCII digit byte
//   name (suffix)             a token                   int32 length + bytes
//
// so the grammar below is written once against this interface. Binary numbers
// are in the byte order the header's `arith` names; `parse_nl` refuses a
// non-host one.
class Tokenizer {
public:
    explicit Tokenizer(std::string_view text) : text_(text) {}

    /// Read every segment item as binary from here on (the header stays text).
    void set_binary() { binary_ = true; }

    // Peek the next non-space character without consuming the token.
    bool peek_char(char& out) {
        skip_ws();
        if (pos_ >= text_.size()) {
            return false;
        }
        out = text_[pos_];
        return true;
    }

    /// Bytes not yet consumed.
    [[nodiscard]] size_t remaining() const { return text_.size() - pos_; }

    bool eof() {
        if (!binary_) {
            skip_ws();
        }
        return pos_ >= text_.size();
    }

    // Read the rest of the current physical line (no leading skip).
    std::string read_line() {
        size_t start = pos_;
        while (pos_ < text_.size() && text_[pos_] != '\n') {
            ++pos_;
        }
        std::string_view line = text_.substr(start, pos_ - start);
        if (pos_ < text_.size()) {
            ++pos_;  // consume newline
        }
        return std::string(line);
    }

    /// A segment marker or expression tag: one character.
    char next_tag() {
        if (!binary_) {
            skip_ws();
        }
        need(1);
        return text_[pos_++];
    }

    /// A number glued to the tag just read ("C3", "o5", "n2.5"). In text no
    /// whitespace may come between, so a tag missing its number is refused
    /// instead of taking the next line's.
    int64_t attached_int(const char* what) {
        if (binary_) {
            return read_binary<int32_t>();
        }
        return to_int(attached_token(what));
    }
    double attached_double(const char* what) {
        if (binary_) {
            return read_binary<double>();
        }
        return to_double(attached_token(what));
    }
    /// The 's' (short) constant: an int16 in binary, plain digits in text.
    int64_t attached_short(const char* what) {
        if (binary_) {
            return read_binary<int16_t>();
        }
        return to_int(attached_token(what));
    }

    int64_t next_int() {
        if (binary_) {
            return read_binary<int32_t>();
        }
        return to_int(next_token());
    }

    double next_double() {
        if (binary_) {
            return read_binary<double>();
        }
        return to_double(next_token());
    }

    /// A bound record's type code: a digit in text, an ASCII digit byte in binary.
    int64_t next_bound_type() {
        if (!binary_) {
            return next_int();
        }
        need(1);
        return static_cast<int64_t>(text_[pos_++]) - '0';
    }

    /// A name (suffix): a token in text, an int32 length then bytes in binary.
    std::string next_name() {
        if (!binary_) {
            return next_token();
        }
        const auto len = read_binary<int32_t>();
        if (len < 0) {
            throw std::runtime_error("NL: negative name length");
        }
        need(static_cast<size_t>(len));
        std::string out(text_.substr(pos_, static_cast<size_t>(len)));
        pos_ += static_cast<size_t>(len);
        return out;
    }

private:
    // Consume one whitespace-delimited token as a string (text only).
    std::string next_token() {
        skip_ws();
        return token_here("token");
    }

    std::string attached_token(const char* what) {
        if (pos_ >= text_.size() || std::isspace(static_cast<unsigned char>(text_[pos_])) != 0) {
            throw std::runtime_error(std::string("NL: ") + what + " is missing its number");
        }
        return token_here(what);
    }

    std::string token_here(const char* what) {
        size_t start = pos_;
        while (pos_ < text_.size() && std::isspace(static_cast<unsigned char>(text_[pos_])) == 0) {
            ++pos_;
        }
        if (start == pos_) {
            throw std::runtime_error(std::string("NL: unexpected end of file while reading ") +
                                     what);
        }
        return std::string(text_.substr(start, pos_ - start));
    }

    // The whole token must be the number: "3x" is refused, not read as 3.
    static int64_t to_int(const std::string& t) {
        size_t used = 0;
        int64_t v = 0;
        try {
            v = std::stoll(t, &used);
        } catch (const std::exception&) {
            throw std::runtime_error("NL: expected integer, got '" + t + "'");
        }
        if (used != t.size()) {
            throw std::runtime_error("NL: expected integer, got '" + t + "'");
        }
        return v;
    }

    static double to_double(const std::string& t) {
        size_t used = 0;
        double v = 0.0;
        try {
            v = std::stod(t, &used);
        } catch (const std::exception&) {
            throw std::runtime_error("NL: expected number, got '" + t + "'");
        }
        if (used != t.size()) {
            throw std::runtime_error("NL: expected number, got '" + t + "'");
        }
        return v;
    }

    void need(size_t n) const {
        if (text_.size() - pos_ < n) {
            throw std::runtime_error("NL: unexpected end of file");
        }
    }

    // One native-order binary value. memcpy, not a cast: the cursor is not
    // aligned for T.
    template <typename T>
    T read_binary() {
        need(sizeof(T));
        T v{};
        std::memcpy(&v, text_.data() + pos_, sizeof(T));
        pos_ += sizeof(T);
        return v;
    }

    // Skip whitespace and '#'-to-EOL comments, safe because '#' never occurs
    // inside a text token. Text only: never called on binary segments.
    void skip_ws() {
        while (pos_ < text_.size()) {
            char c = text_[pos_];
            if (std::isspace(static_cast<unsigned char>(c)) != 0) {
                ++pos_;
            } else if (c == '#') {
                while (pos_ < text_.size() && text_[pos_] != '\n') {
                    ++pos_;
                }
            } else {
                break;
            }
        }
    }

    std::string_view text_;
    size_t pos_ = 0;
    bool binary_ = false;
};

// AMPL opcode -> operand count (ASL `opcode.hd`): 1, 2, 3, or 0 for nullary
// constant ops; n-ary opcodes are is_nary()'s. Anything else returns
// kUnknownArity and the parser throws an "NL_UNKNOWN_OPCODE" error, which the
// MINLPLib runner buckets as unsupported (benchmarks/minlplib/nl_refusal.h).
constexpr int kUnknownArity = -99;

int op_arity(int opcode) {
    switch (opcode) {
        // Binary arithmetic / relations.
        case 0:   // OPPLUS
        case 1:   // OPMINUS
        case 2:   // OPMULT
        case 3:   // OPDIV
        case 4:   // OPREM
        case 5:   // OPPOW
        case 6:   // OPLESS
        case 22:  // LT
        case 23:  // LE
        case 24:  // EQ
        case 28:  // GE
        case 29:  // GT
        case 30:  // NE
        case 48:  // OP_atan2
        case 55:  // OPintDIV
        case 56:  // OPprecision
        case 57:  // OPround
        case 58:  // OPtrunc
        case 73:  // OP_IFF
        case 76:  // OP1POW   (base ^ constant exponent)
        case 78:  // OPCPOW   (constant ^ exponent)
            return 2;
        // Ternary.
        case 35:  // OPIFnl
        case 72:  // OPIMPELSE
            return 3;
        // Unary.
        case 13:  // FLOOR
        case 14:  // CEIL
        case 15:  // ABS
        case 16:  // OPUMINUS
        case 34:  // OPNOT
        case 37:  // OP_tanh
        case 38:  // OP_tan
        case 39:  // OP_sqrt
        case 40:  // OP_sinh
        case 41:  // OP_sin
        case 42:  // OP_log10
        case 43:  // OP_log
        case 44:  // OP_exp
        case 45:  // OP_cosh
        case 46:  // OP_cos
        case 47:  // OP_atanh
        case 49:  // OP_atan
        case 50:  // OP_asinh
        case 51:  // OP_asin
        case 52:  // OP_acosh
        case 53:  // OP_acos
        case 77:  // OP2POW   (x ^ 2)
            return 1;
        default:
            return kUnknownArity;
    }
}

// n-ary opcodes: the operand count appears as the next integer in the stream.
bool is_nary(int opcode) {
    switch (opcode) {
        case 11:  // MINLIST
        case 12:  // MAXLIST
        case 20:  // OPOR
        case 21:  // OPAND
        case 54:  // OPSUMLIST
        case 59:  // OPCOUNT
        case 60:  // OPNUMBEROF
        case 70:  // ANDLIST
        case 71:  // ORLIST
            return true;
        default:
            return false;
    }
}

// An int read from the file, as an int32 index; one out of range is refused
// rather than wrapped.
int32_t to_index(int64_t v, const char* what) {
    if (v < 0 || v > std::numeric_limits<int32_t>::max()) {
        throw std::runtime_error(std::string("NL: ") + what + " " + std::to_string(v) +
                                 " out of range");
    }
    return static_cast<int32_t>(v);
}

// Parse one expression rooted at the current token into `expr.nodes`; return the
// new node's index. Recursive descent over the prefix opcode stream.
int32_t parse_expr(Tokenizer& tok, NlExpr& expr) {
    const char tag = tok.next_tag();

    auto add = [&expr](NlExprNode n) -> int32_t {
        expr.nodes.push_back(std::move(n));
        return static_cast<int32_t>(expr.nodes.size() - 1);
    };

    switch (tag) {
        case 'n':    // numeric constant
        case 's':    // short integer constant (binary writers)
        case 'l': {  // long integer constant (binary writers)
            NlExprNode n;
            n.kind = NlNodeKind::Num;
            if (tag == 'n') {
                n.num = tok.attached_double("constant 'n'");
            } else if (tag == 's') {
                n.num = static_cast<double>(tok.attached_short("constant 's'"));
            } else {
                n.num = static_cast<double>(tok.attached_int("constant 'l'"));
            }
            return add(std::move(n));
        }
        case 'v': {  // variable reference
            NlExprNode n;
            n.kind = NlNodeKind::Var;
            n.index = to_index(tok.attached_int("variable reference 'v'"), "variable reference");
            return add(std::move(n));
        }
        case 'o': {  // operator
            const int opcode = to_index(tok.attached_int("operator 'o'"), "opcode");
            NlExprNode n;
            n.kind = NlNodeKind::Op;
            n.opcode = opcode;

            int arity = op_arity(opcode);
            if (is_nary(opcode)) {
                int64_t count = tok.next_int();
                for (int64_t i = 0; i < count; ++i) {
                    n.children.push_back(parse_expr(tok, expr));
                }
            } else if (arity >= 1 && arity <= 3) {
                for (int i = 0; i < arity; ++i) {
                    n.children.push_back(parse_expr(tok, expr));
                }
            } else {
                // Unknown arity: the operand count is unknowable, so the
                // stream cannot be parsed further; throw the marker.
                throw std::runtime_error("NL_UNKNOWN_OPCODE:" + std::to_string(opcode));
            }
            // Children are indices, not pointers, so they stay valid as
            // `expr.nodes` grows during recursion.
            return add(std::move(n));
        }
        default:
            throw std::runtime_error(std::string("NL: unexpected expression tag '") + tag + "'");
    }
}

// Parse a (possibly nonlinear) segment expression into `expr`, setting its root.
void parse_segment_expr(Tokenizer& tok, NlExpr& expr) {
    expr.root = parse_expr(tok, expr);
}

// The first non-space char of a segment line. Header lines all start with a
// digit, '-' or space, so the first of these letters ends the header.
bool is_segment_marker(char c) {
    switch (c) {
        case 'C':
        case 'O':
        case 'x':
        case 'r':
        case 'b':
        case 'k':
        case 'J':
        case 'G':
        case 'd':
        case 'e':
        case 'f':
        case 'l':
        case 'u':
        case 'V':
        case 'F':
        case 'S':
            return true;
        default:
            return false;
    }
}

// Read up to `count` integers off a header line, zero-filling a short line.
std::vector<int64_t> read_ints(const std::string& line, int count) {
    std::vector<int64_t> out(static_cast<size_t>(count), 0);
    std::istringstream ss(line);
    for (int k = 0; k < count && (ss >> out[static_cast<size_t>(k)]); ++k) {
    }
    return out;
}

// Refuse header counts the input cannot back, before anything is sized from
// them (#289): each variable, constraint, objective and defined variable owns
// at least one payload byte (its `b` record, `C`, `O`, `V` segment), so their
// sum cannot exceed the bytes left, else `resize` could ask for tens of GB.
// Each count is int32 and, once checked, non-negative: the int64 sum cannot
// overflow.
void check_counts_backed(const NlProblem& prob, size_t remaining) {
    if (prob.n_vars < 0 || prob.n_cons < 0 || prob.n_objs < 0) {
        throw std::runtime_error(
            "NL: malformed header: negative variable, constraint or "
            "objective count");
    }
    const int64_t declared = int64_t{prob.n_vars} + prob.n_cons + prob.n_objs + prob.n_defined_vars;
    if (static_cast<uint64_t>(declared) > remaining) {
        throw std::runtime_error("NL: malformed header: " + std::to_string(declared) +
                                 " declared items (variables, constraints, objectives, defined "
                                 "variables) but only " +
                                 std::to_string(remaining) + " bytes follow the header");
    }
}

// The header counts needed to place the discrete columns. Everything else in
// the header is consumed and discarded.
struct NlHeaderCounts {
    int64_t nlvc = 0;     ///< nonlinear in constraints (index bound)
    int64_t nlvo = 0;     ///< nonlinear in objectives (index bound)
    int64_t nlvb = 0;     ///< nonlinear in both
    int64_t nbv = 0;      ///< linear binary
    int64_t niv = 0;      ///< linear integer
    int64_t nlvbi = 0;    ///< integer among the nonlinear-in-both block
    int64_t nlvci = 0;    ///< integer among the nonlinear-in-constraints block
    int64_t nlvoi = 0;    ///< integer among the nonlinear-in-objectives block
    int64_t arith = 0;    ///< binary byte order: 1 little-endian, 2 big-endian IEEE
    bool binary = false;  ///< 'b' header: segments are binary
};

// Consume the whole NL header, filling `prob`'s counts and returning the
// discrete-placement counts. Leaves the cursor on the first segment marker.
NlHeaderCounts parse_header(Tokenizer& tok, NlProblem& prob) {
    // Line 1: format char + version, e.g. "g3 0 1 0".
    char fmt = 0;
    if (!tok.peek_char(fmt)) {
        throw std::runtime_error("NL: empty file");
    }
    if (fmt != 'g' && fmt != 'b') {
        throw std::runtime_error(std::string("NL: unrecognised header char '") + fmt +
                                 "' (expected 'g' for text or 'b' for binary format)");
    }
    // Must consume line 1, else the counts below are read from it.
    tok.read_line();

    // Line 2: " nvar ncon nobj nranges neqn ..." (counts). We need the first 3.
    std::string line2 = tok.read_line();
    {
        std::istringstream ss(line2);
        ss >> prob.n_vars >> prob.n_cons >> prob.n_objs;
        if (!ss) {
            throw std::runtime_error("NL: malformed counts line: '" + line2 + "'");
        }
    }

    // The header-line count varies by version, so consume lines (each starts
    // with a digit, '-' or space) until a segment marker letter. Kept, counting
    // from line 3:
    //   #3 -> header line 5: "nlvc nlvo nlvb"          (nonlinear var counts)
    //   #4 -> header line 6: "nwv nfunc arith flags"; only `arith` (binary
    //    byte order). Arc and other-linear columns are continuous and precede
    //    the nbv+niv block, so they never shift a discrete position.
    //   #5 -> header line 7: "nbv niv nlvbi nlvci nlvoi"  (discrete counts)
    //   #8 -> header line 10: "comb comc como comc1 como1" (defined variables)
    NlHeaderCounts h;
    h.binary = fmt == 'b';
    int header_line_after_counts = 0;
    while (true) {
        char c = 0;
        if (!tok.peek_char(c)) {
            break;  // no segments (degenerate, e.g. counts-only fixture)
        }
        if (is_segment_marker(c)) {
            break;
        }
        std::string hline = tok.read_line();  // another header line
        ++header_line_after_counts;
        if (header_line_after_counts == 3) {
            auto v = read_ints(hline, 3);
            h.nlvc = v[0];
            h.nlvo = v[1];
            h.nlvb = v[2];
        } else if (header_line_after_counts == 4) {
            h.arith = read_ints(hline, 3)[2];
        } else if (header_line_after_counts == 5) {
            auto v = read_ints(hline, 5);
            h.nbv = v[0];
            h.niv = v[1];
            h.nlvbi = v[2];
            h.nlvci = v[3];
            h.nlvoi = v[4];
            // `prob.n_discrete_vars` is set by `mark_discrete_vars` after the
            // counts are range-checked: summed here unchecked it could wrap.
        } else if (header_line_after_counts == 8) {
            const auto v = read_ints(hline, 5);
            // Each within [0, INT32_MAX] first, so the int64 sum cannot wrap.
            constexpr int64_t kMax = std::numeric_limits<int32_t>::max();
            if (std::any_of(v.begin(), v.end(), [](int64_t n) { return n < 0 || n > kMax; })) {
                throw std::runtime_error("NL: malformed common-expression line: '" + hline + "'");
            }
            const int64_t total = v[0] + v[1] + v[2] + v[3] + v[4];
            if (total > kMax) {
                throw std::runtime_error("NL: malformed common-expression line: '" + hline + "'");
            }
            prob.n_defined_vars = static_cast<int32_t>(total);
        }
    }
    return h;
}

// Refuse placement counts (header lines 5 and 7) outside [0, n_vars] and return
// the declared discrete count (#290), else a huge block length hangs
// `mark_tail` and the sums below wrap int64. None exceeds n_vars in a valid
// file (column counts, or for nlvc/nlvo an index bound). `n_vars` is a
// non-negative int32, so after the check any sum of five fits int64 and the
// discrete total fits int32.
int32_t checked_discrete_count(const NlProblem& prob, const NlHeaderCounts& h) {
    const int64_t n = prob.n_vars;
    const std::array<std::pair<const char*, int64_t>, 8> counts = {{{"nlvc", h.nlvc},
                                                                    {"nlvo", h.nlvo},
                                                                    {"nlvb", h.nlvb},
                                                                    {"nbv", h.nbv},
                                                                    {"niv", h.niv},
                                                                    {"nlvbi", h.nlvbi},
                                                                    {"nlvci", h.nlvci},
                                                                    {"nlvoi", h.nlvoi}}};
    for (const auto& [field, c] : counts) {
        if (c < 0 || c > n) {
            throw std::runtime_error("NL: malformed header: " + std::string(field) + " = " +
                                     std::to_string(c) + " outside [0, " + std::to_string(n) +
                                     "] (the variable count)");
        }
    }
    const int64_t discrete = h.nbv + h.niv + h.nlvbi + h.nlvci + h.nlvoi;
    if (discrete > n) {
        throw std::runtime_error("NL: malformed header: " + std::to_string(discrete) +
                                 " discrete variables declared but only " + std::to_string(n) +
                                 " variables");
    }
    return static_cast<int32_t>(discrete);
}

// Refuse block counts violating the relations Gay's ordering implies (#291;
// see `mark_discrete_vars`), else e.g. nlvc=2, nlvb=3, nlvbi=1 passes the
// placement self-check with a guessed integer column. Must follow
// `checked_discrete_count`, so no sum or difference here overflows.
void check_block_relations(const NlProblem& prob, const NlHeaderCounts& h) {
    const auto refuse = [](const std::string& what) {
        throw std::runtime_error("NL: malformed header: " + what +
                                 " (inconsistent variable-block counts)");
    };
    const auto str = [](int64_t v) { return std::to_string(v); };
    // Variables nonlinear in both are a subset of those nonlinear in constraints.
    if (h.nlvb > h.nlvc) {
        refuse("nlvb = " + str(h.nlvb) + " exceeds nlvc = " + str(h.nlvc));
    }
    // Each block's integer count fits in that block.
    if (h.nlvbi > h.nlvb) {
        refuse("nlvbi = " + str(h.nlvbi) + " exceeds nlvb = " + str(h.nlvb));
    }
    if (h.nlvci > h.nlvc - h.nlvb) {
        refuse("nlvci = " + str(h.nlvci) + " exceeds nlvc - nlvb = " + str(h.nlvc - h.nlvb));
    }
    const int64_t obj_only = std::max<int64_t>(h.nlvo - h.nlvc, 0);
    if (h.nlvoi > obj_only) {
        refuse("nlvoi = " + str(h.nlvoi) + " exceeds max(nlvo - nlvc, 0) = " + str(obj_only));
    }
    // The nonlinear blocks span [0, max(nlvc, nlvo)); the linear discrete
    // block is the last nbv + niv columns. They must not overlap.
    const int64_t nonlinear = std::max(h.nlvc, h.nlvo);
    if (nonlinear + h.nbv + h.niv > prob.n_vars) {
        refuse("max(nlvc, nlvo) + nbv + niv = " + str(nonlinear + h.nbv + h.niv) + " exceeds " +
               str(prob.n_vars) + " variables");
    }
}

// The last `n_int` columns of [block_start, block_start + block_len) are integer.
// `checked_discrete_count` keeps the sums from overflowing; the clip to
// [0, n_vars) bounds the loop should a caller skip it.
void mark_tail(NlProblem& prob, int64_t block_start, int64_t block_len, int64_t n_int) {
    const int64_t end = std::min<int64_t>(block_start + block_len, prob.n_vars);
    const int64_t begin = std::max({block_start + block_len - n_int, block_start, int64_t{0}});
    for (int64_t j = begin; j < end; ++j) {
        prob.var_is_discrete[static_cast<size_t>(j)] = 1;
    }
}

// Integer variable *positions* follow Gay's variable ordering ("Hooking Your
// Solver to AMPL", the variable-order table). Columns are laid out as:
//
//   1. nonlinear in both constraints and objectives   nlvb        (last nlvbi integer)
//   2. nonlinear in constraints only                  nlvc - nlvb (last nlvci integer)
//   3. nonlinear in objectives only                   nlvo - nlvc (last nlvoi integer)
//   4. linear arc variables                           nwv         (continuous)
//   5. other linear                                   remainder   (continuous)
//   6. binary                                         nbv
//   7. other integer                                  niv
//
// so each nonlinear block's integers trail it, and the linear discrete columns
// are the last nbv+niv.
void mark_discrete_vars(NlProblem& prob, const NlHeaderCounts& h) {
    prob.n_discrete_vars = checked_discrete_count(prob, h);  // refuses before any loop
    check_block_relations(prob, h);
    prob.var_is_discrete.assign(static_cast<size_t>(prob.n_vars), 0);
    const int64_t cat1 = h.nlvb;                                 // nonlinear in both
    const int64_t cat2 = std::max<int64_t>(h.nlvc - h.nlvb, 0);  // nonlinear in constraints only
    // Objective-only block [nlvc, nlvo), empty when nlvo <= nlvc: `nlvo` is an
    // index bound (nonlinear columns total max(nlvc, nlvo)). Must not be
    // `nlvo - nlvb`, which overshoots by nlvc - nlvb (windfac: 24 of 14 columns).
    const int64_t cat3 = std::max<int64_t>(h.nlvo - (cat1 + cat2), 0);
    mark_tail(prob, 0, cat1, h.nlvbi);
    mark_tail(prob, cat1, cat2, h.nlvci);
    mark_tail(prob, cat1 + cat2, cat3, h.nlvoi);
    // Trailing linear discrete block: the final nbv + niv columns.
    mark_tail(prob, 0, prob.n_vars, h.nbv + h.niv);

    // Backstop to `check_block_relations`: placed positions must equal the
    // header's count, else refuse rather than build wrong integrality.
    int32_t marked = 0;
    for (uint8_t f : prob.var_is_discrete) {
        marked += f;
    }
    if (marked != prob.n_discrete_vars) {
        throw std::runtime_error(
            "NL: discrete-variable placement disagrees with the header count (placed " +
            std::to_string(marked) + ", header declares " + std::to_string(prob.n_discrete_vars) +
            ") — unexpected variable ordering");
    }
}

// One `<type> [values]` bound record. The `r` (constraint) and `b` (variable)
// segments carry the identical grammar; only the struct it lands in differs.
struct BoundRecord {
    NlBoundType type = NlBoundType::Free;
    double lower = -kNlInf;
    double upper = kNlInf;
};

// Segment payload readers, one grammar each, so the dispatch below is a table.

// `x`: initial primal guess — `count` pairs of <varidx> <value>.
void read_initial_guess(Tokenizer& tok, NlProblem& prob, int64_t count) {
    for (int64_t k = 0; k < count; ++k) {
        int64_t vi = tok.next_int();
        double val = tok.next_double();
        if (vi >= 0 && vi < prob.n_vars) {
            prob.initial_x[vi] = val;
        }
    }
}

// `k`: Jacobian column counts. Unused (J terms are stored sparse), but must be
// consumed to keep the cursor aligned.
void skip_ints(Tokenizer& tok, int64_t count) {
    for (int64_t k = 0; k < count; ++k) {
        tok.next_int();
    }
}

// `J` / `G`: `k` pairs of <varidx> <coef>, appended to a constraint's or an
// objective's linear part.
void read_linear_terms(Tokenizer& tok, int64_t k, std::vector<NlLinTerm>& out) {
    for (int64_t t = 0; t < k; ++t) {
        NlLinTerm term;
        term.var = static_cast<int32_t>(tok.next_int());
        term.coef = tok.next_double();
        out.push_back(term);
    }
}

// `V`: defined variable <i> (an index past the columns) "<k> <l>", then k
// linear terms and one expression. `l` says where the variable is used, which
// does not change its value.
void read_defined_var(Tokenizer& tok, NlProblem& prob, int64_t i) {
    const int64_t k = tok.next_int();
    tok.next_int();  // l
    const int64_t slot = i - prob.n_vars;
    if (slot < 0 || slot >= prob.n_defined_vars) {
        throw std::runtime_error("NL: V-segment index " + std::to_string(i) +
                                 " is outside the header's defined-variable range");
    }
    NlDefinedVar& dv = prob.defined_vars[static_cast<size_t>(slot)];
    if (dv.defined) {
        throw std::runtime_error("NL: defined variable " + std::to_string(i) + " defined twice");
    }
    if (k < 0) {
        throw std::runtime_error("NL: V-segment with a negative linear-term count");
    }
    read_linear_terms(tok, k, dv.linear);
    parse_segment_expr(tok, dv.nonlinear);
    dv.defined = true;
}

// `d`: initial dual guess, `count` pairs of <conidx> <value>. Not needed by a
// primal search; consumed to keep the cursor aligned.
void skip_dual_guess(Tokenizer& tok, int64_t count) {
    for (int64_t k = 0; k < count; ++k) {
        tok.next_int();
        tok.next_double();
    }
}

// The index glued to a segment marker, e.g. 3 in "J3".
int64_t segment_index(Tokenizer& tok, char kind) {
    const std::string what = std::string("segment '") + kind + "'";
    return tok.attached_int(what.c_str());
}

BoundRecord read_bound(Tokenizer& tok) {
    BoundRecord b;
    const int64_t code = tok.next_bound_type();
    // Type 5 (complementarity, `5 k i`) and anything outside 0..4 must be
    // refused before the cast, else the cursor stops mid-record and the rest is
    // misparsed (#211). "Unsupported" wording: the MINLPLib runner buckets it
    // as a coverage gap, not a broken file.
    if (code < 0 || code > static_cast<int64_t>(NlBoundType::Equal)) {
        throw std::runtime_error("NL: bound type " + std::to_string(code) +
                                 (code == 5 ? " (complementarity)" : "") +
                                 " is not supported by this reader");
    }
    b.type = static_cast<NlBoundType>(code);
    switch (b.type) {
        case NlBoundType::Range:
            b.lower = tok.next_double();
            b.upper = tok.next_double();
            break;
        case NlBoundType::Upper:
            b.upper = tok.next_double();
            b.lower = -kNlInf;
            break;
        case NlBoundType::Lower:
            b.lower = tok.next_double();
            b.upper = kNlInf;
            break;
        case NlBoundType::Free:
            b.lower = -kNlInf;
            b.upper = kNlInf;
            break;
        case NlBoundType::Equal:
            b.lower = b.upper = tok.next_double();
            break;
    }
    return b;
}

// `r`: one bound record per constraint, in constraint order.
void read_con_bounds(Tokenizer& tok, NlProblem& prob) {
    for (int64_t i = 0; i < prob.n_cons; ++i) {
        const BoundRecord rec = read_bound(tok);
        NlConBound& b = prob.constraints[i].bound;
        b.type = rec.type;
        b.lower = rec.lower;
        b.upper = rec.upper;
    }
}

// `b`: one bound record per variable, in column order.
void read_var_bounds(Tokenizer& tok, NlProblem& prob) {
    for (int64_t i = 0; i < prob.n_vars; ++i) {
        const BoundRecord rec = read_bound(tok);
        NlVarBound& vb = prob.var_bounds[i];
        vb.type = rec.type;
        vb.lower = rec.lower;
        vb.upper = rec.upper;
    }
}

// Dispatch one segment, named by its tag `kind` (the letter of "C0", "O0", "r",
// "J3"; any index follows in the stream), onto the reader for its payload.
void parse_segment(Tokenizer& tok, NlProblem& prob, char kind) {
    switch (kind) {
        case 'C': {  // nonlinear part of constraint <i>
            const int64_t i = segment_index(tok, kind);
            if (i < 0 || i >= prob.n_cons) {
                throw std::runtime_error("NL: C-segment index out of range");
            }
            parse_segment_expr(tok, prob.constraints[i].nonlinear);
            break;
        }
        case 'O': {  // objective <i> <sense>; then nonlinear expr
            const int64_t i = segment_index(tok, kind);
            if (i < 0 || i >= prob.n_objs) {
                throw std::runtime_error("NL: O-segment index out of range");
            }
            const int64_t sense = tok.next_int();  // 0 min, 1 max
            prob.objectives[i].maximize = (sense != 0);
            parse_segment_expr(tok, prob.objectives[i].nonlinear);
            break;
        }
        case 'x':  // initial primal guess: count, then <varidx> <value> pairs
            read_initial_guess(tok, prob, segment_index(tok, kind));
            break;
        case 'r':  // constraint bounds: n_cons records of <type> [values]
            read_con_bounds(tok, prob);
            break;
        case 'b':  // variable bounds: n_vars records of <type> [values]
            read_var_bounds(tok, prob);
            break;
        case 'k':  // Jacobian column-count header: n_vars-1 cumulative ints
            skip_ints(tok, segment_index(tok, kind));
            break;
        case 'J': {  // linear part of constraint <i>: k pairs <varidx> <coef>
            const int64_t i = segment_index(tok, kind);
            const int64_t k = tok.next_int();
            if (i < 0 || i >= prob.n_cons) {
                throw std::runtime_error("NL: J-segment index out of range");
            }
            read_linear_terms(tok, k, prob.constraints[i].linear);
            break;
        }
        case 'G': {  // linear part of objective <i>: k pairs <varidx> <coef>
            const int64_t i = segment_index(tok, kind);
            const int64_t k = tok.next_int();
            if (i < 0 || i >= prob.n_objs) {
                throw std::runtime_error("NL: G-segment index out of range");
            }
            read_linear_terms(tok, k, prob.objectives[i].linear);
            break;
        }
        case 'd':  // dual initial guess: count then pairs
            skip_dual_guess(tok, segment_index(tok, kind));
            break;
        case 'V':  // defined variable: index then linear+nonlinear def
            read_defined_var(tok, prob, segment_index(tok, kind));
            break;
        case 'S': {  // suffix block: "S<kind> <n> <name>"
            // Refused, not skipped: a suffix can be model data (MINLPLib's
            // `sosno`/`ref` declare SOS; dropping them solves a relaxation).
            // Named so a coverage list says which suffix stopped the file.
            segment_index(tok, kind);  // the suffix kind
            tok.next_int();            // its entry count
            const std::string suffix = tok.next_name();
            throw std::runtime_error("NL: suffix '" + suffix +
                                     "' (segment 'S') is not supported by this reader");
        }
        case 'F':  // imported function declaration
            // An imported (external) function has no DAG op to map onto.
            throw std::runtime_error(std::string("NL: segment '") + kind +
                                     "' is not supported by this reader");
        default:
            throw std::runtime_error(std::string("NL: unknown segment marker '") + kind + "'");
    }
}

// The header `arith` code for this host's byte order (ASL: 1 = IEEE
// little-endian, 2 = IEEE big-endian).
int64_t host_arith() {
    const uint16_t probe = 1;
    unsigned char first = 0;
    std::memcpy(&first, &probe, 1);
    return first == 1 ? 1 : 2;
}

}  // namespace

NlProblem parse_nl(const std::string& text, const std::string& name) {
    NlProblem prob;
    prob.name = name;
    Tokenizer tok(text);

    // ---- Header ----
    const NlHeaderCounts header = parse_header(tok, prob);

    // Byte order first: a foreign-order file is well formed, so it must not be
    // reported as a malformed header. The size check precedes every
    // count-sized vector.
    if (header.binary) {
        // A foreign or undeclared order would parse as garbage, so refuse,
        // worded as a coverage gap so the MINLPLib runner buckets it as skipped.
        const int64_t host = host_arith();
        if (header.arith != host) {
            throw std::runtime_error("NL: binary file's byte order (arith " +
                                     std::to_string(header.arith) + ", this host's is arith " +
                                     std::to_string(host) + ") is not supported by this reader");
        }
        tok.set_binary();
    }
    check_counts_backed(prob, tok.remaining());
    mark_discrete_vars(prob, header);

    prob.constraints.resize(prob.n_cons);
    prob.objectives.resize(prob.n_objs);
    prob.var_bounds.assign(prob.n_vars, NlVarBound{});
    prob.initial_x.assign(prob.n_vars, std::numeric_limits<double>::quiet_NaN());
    prob.defined_vars.resize(static_cast<size_t>(prob.n_defined_vars));

    // ---- Segments ----
    while (!tok.eof()) {
        // A named local: `tok.next_tag()` as an argument beside `tok` would be
        // unsequenced against the other arguments' initialisation.
        const char kind = tok.next_tag();
        parse_segment(tok, prob, kind);
    }

    return prob;
}

NlProblem read_nl(const std::string& filename) {
    std::ifstream f(filename, std::ios::binary);
    if (!f) {
        throw std::runtime_error("NL: cannot open file: " + filename);
    }
    std::ostringstream buf;
    buf << f.rdbuf();
    return parse_nl(buf.str(), filename);
}

}  // namespace cbls
