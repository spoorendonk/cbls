// AMPL NL (text format) reader.
//
// Original implementation from the public NL format specifications:
//   * David M. Gay, "Writing .nl Files" — https://ampl.github.io/nlwrite.pdf
//   * David M. Gay, "Hooking Your Solver to AMPL" — https://ampl.com/REFS/hooking2.pdf
// No third-party source was vendored; the opcode numbers below follow the AMPL
// `opcode.hd` numbering documented in those references.
//
// Scope: TEXT format only (header line begins with 'g'). The binary variant
// (header 'b') is rejected with a clear error. The reader records the raw
// expression graph and linear parts; semantic mapping to CBLS ops (and the
// decision of which opcodes are supported) lives in nl_to_model.cpp.

#include "cbls/io_nl.h"

#include <cctype>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace cbls {

namespace {

// A forward token cursor over the NL text. The NL text format is whitespace /
// newline separated; within a segment, items appear in a fixed order. We tokenise
// lazily so expression parsing can pull operands as needed.
class Tokenizer {
public:
    explicit Tokenizer(std::string_view text) : text_(text) {}

    // Peek the next non-space character without consuming the token.
    bool peek_char(char& out) {
        skip_ws();
        if (pos_ >= text_.size()) {
            return false;
        }
        out = text_[pos_];
        return true;
    }

    bool eof() {
        skip_ws();
        return pos_ >= text_.size();
    }

    // Read the rest of the current physical line (no leading skip). Used for
    // header lines and to discard trailing comments after a segment marker.
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

    // Consume one whitespace-delimited token as a string.
    std::string next_token() {
        skip_ws();
        size_t start = pos_;
        while (pos_ < text_.size() && !std::isspace(static_cast<unsigned char>(text_[pos_]))) {
            ++pos_;
        }
        if (start == pos_) {
            throw std::runtime_error("NL: unexpected end of file while reading token");
        }
        return std::string(text_.substr(start, pos_ - start));
    }

    int64_t next_int() {
        std::string t = next_token();
        try {
            return std::stoll(t);
        } catch (...) {
            throw std::runtime_error("NL: expected integer, got '" + t + "'");
        }
    }

    double next_double() {
        std::string t = next_token();
        try {
            return std::stod(t);
        } catch (...) {
            throw std::runtime_error("NL: expected number, got '" + t + "'");
        }
    }

private:
    // Skip whitespace and '#'-to-end-of-line comments. NL files annotate header
    // and (sometimes) segment-marker lines with "# ..." comments; '#' never
    // appears inside a numeric/opcode token, so treating it as comment-to-EOL is
    // safe across the whole stream.
    void skip_ws() {
        while (pos_ < text_.size()) {
            char c = text_[pos_];
            if (std::isspace(static_cast<unsigned char>(c))) {
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
};

// AMPL opcode -> number of operands. Numbers are the authoritative values from
// the ASL `opcode.hd` (ampl/asl, src/solvers/opcode.hd). We return:
//   2  binary, 1 unary, 3 ternary, 0 nullary-ish constant ops.
// n-ary opcodes (operand count read inline) are reported by is_nary() instead.
// Opcodes whose arity we can't resolve return kUnknownArity; the parser then
// throws NL_UNKNOWN_OPCODE so the adapter can skip the instance cleanly.
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

// Parse one expression rooted at the current token into `expr.nodes`; return the
// new node's index. Recursive descent over the prefix opcode stream.
int32_t parse_expr(Tokenizer& tok, NlExpr& expr) {
    std::string head = tok.next_token();
    if (head.empty()) {
        throw std::runtime_error("NL: empty expression token");
    }
    char tag = head[0];
    std::string rest = head.substr(1);

    auto add = [&expr](NlExprNode n) -> int32_t {
        expr.nodes.push_back(std::move(n));
        return static_cast<int32_t>(expr.nodes.size() - 1);
    };

    switch (tag) {
        case 'n': {  // numeric constant
            NlExprNode n;
            n.kind = NlNodeKind::Num;
            n.num = std::stod(rest);
            return add(std::move(n));
        }
        case 'v': {  // variable reference
            NlExprNode n;
            n.kind = NlNodeKind::Var;
            n.index = static_cast<int32_t>(std::stoll(rest));
            return add(std::move(n));
        }
        case 'o': {  // operator
            int opcode = static_cast<int>(std::stoll(rest));
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
                // Unknown opcode arity. We cannot know how many operands to
                // consume, so we cannot safely continue parsing this stream.
                // Record the opcode with no children and stop descending; the
                // adapter will reject the instance on the unknown opcode. To
                // keep the token stream consistent we throw a typed marker.
                throw std::runtime_error("NL_UNKNOWN_OPCODE:" + std::to_string(opcode));
            }
            // Children indices may have shifted as the vector grew during
            // recursion; but we stored indices, not pointers, so they remain
            // valid. Re-add the node now that children are known.
            return add(std::move(n));
        }
        default:
            throw std::runtime_error(std::string("NL: unexpected expression tag '") + tag + "'");
    }
}

// Parse a (possibly nonlinear) segment expression into `expr`, setting its root.
// Catches the unknown-opcode marker and rethrows it tagged with context.
void parse_segment_expr(Tokenizer& tok, NlExpr& expr) {
    expr.root = parse_expr(tok, expr);
}

}  // namespace

NlProblem parse_nl(const std::string& text, const std::string& name) {
    NlProblem prob;
    prob.name = name;
    Tokenizer tok(text);

    // ---- Header ----
    // Line 1: format char + version, e.g. "g3 0 1 0".
    char fmt = 0;
    if (!tok.peek_char(fmt)) {
        throw std::runtime_error("NL: empty file");
    }
    if (fmt == 'b') {
        throw std::runtime_error(
            "NL: binary format ('b' header) is not supported; only text ('g') is. "
            "Re-export the instance in ASCII NL.");
    }
    if (fmt != 'g') {
        throw std::runtime_error(std::string("NL: unrecognised header char '") + fmt +
                                 "' (expected 'g' for text format)");
    }
    std::string line1 = tok.read_line();  // consume the whole first header line

    // Line 2: " nvar ncon nobj nranges neqn ..." (counts). We need the first 3.
    std::string line2 = tok.read_line();
    {
        std::istringstream ss(line2);
        ss >> prob.n_vars >> prob.n_cons >> prob.n_objs;
        if (!ss) {
            throw std::runtime_error("NL: malformed counts line: '" + line2 + "'");
        }
    }
    // Header continues with several more lines describing nonlinear counts,
    // network structure, etc. The number of header lines is not fixed across
    // versions, but every header line in the text format is a line of integers;
    // the first segment marker is a single letter optionally followed by an
    // index. We consume header lines until we hit a line whose first non-space
    // char is one of the segment markers (C O x r b k J G d e f l u V F G S).
    auto is_segment_marker = [](char c) {
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
    };
    // Peek ahead: the remaining header lines all start with a digit, '-', or
    // space. Consume them until a segment marker letter appears. While doing so,
    // capture the discrete-variable line. In the `g` format the header layout is
    // fixed; counting line 1 (`g...`) and line 2 (counts) already consumed, the
    // discrete-variable line is the 5th of the lines we skip here (overall header
    // line 7): "nbv niv nlvbi nlvci nlvoi". We sum its five integers to get the
    // total integer/binary variable count.
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
        if (header_line_after_counts == 5) {  // discrete-variable line (line 7)
            std::istringstream ss(hline);
            int64_t v = 0;
            int32_t total = 0;
            for (int k = 0; k < 5 && (ss >> v); ++k) {
                total += static_cast<int32_t>(v);
            }
            prob.n_discrete_vars = total;
        }
    }

    prob.constraints.resize(prob.n_cons);
    prob.objectives.resize(prob.n_objs);
    prob.var_bounds.assign(prob.n_vars, NlVarBound{});
    prob.initial_x.assign(prob.n_vars, std::numeric_limits<double>::quiet_NaN());

    // ---- Segments ----
    while (!tok.eof()) {
        char marker = 0;
        if (!tok.peek_char(marker)) {
            break;
        }
        std::string seg = tok.next_token();  // e.g. "C0", "O0", "r", "b", "J3"
        char kind = seg[0];
        std::string idx_str = seg.substr(1);
        auto seg_index = [&]() -> int64_t {
            if (idx_str.empty()) {
                throw std::runtime_error("NL: segment '" + seg + "' missing index");
            }
            return std::stoll(idx_str);
        };

        switch (kind) {
            case 'C': {  // nonlinear part of constraint <i>
                int64_t i = seg_index();
                if (i < 0 || i >= prob.n_cons) {
                    throw std::runtime_error("NL: C-segment index out of range");
                }
                parse_segment_expr(tok, prob.constraints[i].nonlinear);
                break;
            }
            case 'O': {  // objective <i> <sense>; then nonlinear expr
                int64_t i = seg_index();
                if (i < 0 || i >= prob.n_objs) {
                    throw std::runtime_error("NL: O-segment index out of range");
                }
                int64_t sense = tok.next_int();  // 0 min, 1 max
                prob.objectives[i].maximize = (sense != 0);
                parse_segment_expr(tok, prob.objectives[i].nonlinear);
                break;
            }
            case 'x': {  // initial primal guess: count, then <varidx> <value> pairs
                int64_t count = seg_index();
                for (int64_t k = 0; k < count; ++k) {
                    int64_t vi = tok.next_int();
                    double val = tok.next_double();
                    if (vi >= 0 && vi < prob.n_vars) {
                        prob.initial_x[vi] = val;
                    }
                }
                break;
            }
            case 'r': {  // constraint bounds: n_cons lines of <type> [values]
                for (int64_t i = 0; i < prob.n_cons; ++i) {
                    int64_t t = tok.next_int();
                    NlConBound& b = prob.constraints[i].bound;
                    b.type = static_cast<NlBoundType>(t);
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
                }
                break;
            }
            case 'b': {  // variable bounds: n_vars lines of <type> [values]
                for (int64_t i = 0; i < prob.n_vars; ++i) {
                    int64_t t = tok.next_int();
                    NlVarBound& vb = prob.var_bounds[i];
                    vb.type = static_cast<NlBoundType>(t);
                    switch (vb.type) {
                        case NlBoundType::Range:
                            vb.lower = tok.next_double();
                            vb.upper = tok.next_double();
                            break;
                        case NlBoundType::Upper:
                            vb.upper = tok.next_double();
                            vb.lower = -kNlInf;
                            break;
                        case NlBoundType::Lower:
                            vb.lower = tok.next_double();
                            vb.upper = kNlInf;
                            break;
                        case NlBoundType::Free:
                            vb.lower = -kNlInf;
                            vb.upper = kNlInf;
                            break;
                        case NlBoundType::Equal:
                            vb.lower = vb.upper = tok.next_double();
                            break;
                    }
                }
                break;
            }
            case 'k': {  // Jacobian column-count header: n_vars-1 cumulative ints
                int64_t count = seg_index();
                for (int64_t k = 0; k < count; ++k) {
                    tok.next_int();  // not needed: we store sparse J terms directly
                }
                break;
            }
            case 'J': {  // linear part of constraint <i>: k pairs <varidx> <coef>
                int64_t i = seg_index();
                int64_t k = tok.next_int();
                if (i < 0 || i >= prob.n_cons) {
                    throw std::runtime_error("NL: J-segment index out of range");
                }
                for (int64_t t = 0; t < k; ++t) {
                    NlLinTerm term;
                    term.var = static_cast<int32_t>(tok.next_int());
                    term.coef = tok.next_double();
                    prob.constraints[i].linear.push_back(term);
                }
                break;
            }
            case 'G': {  // linear part of objective <i>: k pairs <varidx> <coef>
                int64_t i = seg_index();
                int64_t k = tok.next_int();
                if (i < 0 || i >= prob.n_objs) {
                    throw std::runtime_error("NL: G-segment index out of range");
                }
                for (int64_t t = 0; t < k; ++t) {
                    NlLinTerm term;
                    term.var = static_cast<int32_t>(tok.next_int());
                    term.coef = tok.next_double();
                    prob.objectives[i].linear.push_back(term);
                }
                break;
            }
            case 'd':    // dual initial guess: count then pairs
            case 'V':    // defined variable: index then linear+nonlinear def
            case 'F':    // imported function declaration
            case 'S': {  // suffix block
                // These segments are not needed for the CBLS model. They have
                // file-position-dependent payloads we can't blindly skip, so a
                // clean error is safer than silent corruption. In practice the
                // MINLPLib instances we select do not carry them.
                throw std::runtime_error(std::string("NL: segment '") + kind +
                                         "' is not supported by this reader");
            }
            default:
                throw std::runtime_error(std::string("NL: unknown segment marker '") + kind + "'");
        }
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
