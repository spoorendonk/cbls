#pragma once
// How a MINLPLib tool tells a coverage gap from a broken file when the NL
// reader throws. The reader words every refusal of a well-formed construct it
// does not implement (an unknown opcode, an SOS suffix, a complementarity
// bound, a foreign byte order) so that one of these markers is in it; the
// runner (`cbls_minlplib`) tallies such a file as skipped(unsupported), and
// `cbls_minlplib_nl_check` as reader-unsupported, by this one test.

#include <string>

namespace cbls::minlplib {

inline bool is_reader_coverage_gap(const std::string& what) {
    return what.find("NL_UNKNOWN_OPCODE") != std::string::npos ||
           what.find("not supported by this reader") != std::string::npos;
}

}  // namespace cbls::minlplib
