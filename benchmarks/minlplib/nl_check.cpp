// MINLPLib NL checker: reads and models each NL file named on the command
// line, without solving, and prints one tab-separated line per file:
//
//   <path> \t modelled
//   <path> \t reader-error \t <message>          the file is malformed
//   <path> \t reader-unsupported \t <message>    a construct the reader lacks
//   <path> \t adapter-unsupported \t <reason>    an op the adapter lacks
//   <path> \t checker-failed \t <message>        the adapter or model threw
//
// With `--presolve` first, the model is built with the linear-part presolve
// (#326, #343) and a modelled line also carries its outcome, for the census:
//
//   <path> \t modelled \t presolve=<status> rows=<removed> cols=<removed> bounds=<tightened>
//
// benchmarks/instances/minlplib/status.py runs it over the unpacked MINLPLib
// archive to give every catalogue instance a status (#283). Reader refusals
// are split as the runner splits them (nl_refusal.h). Every control character
// in a message (reader messages quote raw file bytes) becomes a space, so each
// file is exactly one line.
#include <benchmarks/minlplib/nl_refusal.h>
#include <cbls/io_nl.h>
#include <cstdio>
#include <exception>
#include <string>

namespace {

std::string one_line(std::string s) {
    for (char& c : s) {
        const auto u = static_cast<unsigned char>(c);
        if (u < 0x20 || u == 0x7f) {
            c = ' ';
        }
    }
    return s;
}

const char* status_name(cbls::PresolveStatus s) {
    switch (s) {
        case cbls::PresolveStatus::Unchanged:
            return "unchanged";
        case cbls::PresolveStatus::Reduced:
            return "reduced";
        case cbls::PresolveStatus::Infeasible:
            return "infeasible";
    }
    return "?";
}

std::string presolve_cells(const cbls::PresolveStats& s) {
    const int rows = s.rows_empty + s.rows_free + s.rows_singleton + s.rows_redundant;
    const int cols = s.cols_fixed + s.cols_empty;
    return std::string("\tpresolve=") + status_name(s.status) + " rows=" + std::to_string(rows) +
           " cols=" + std::to_string(cols) + " bounds=" + std::to_string(s.bounds_tightened);
}

std::string check(const std::string& path, bool presolve) {
    cbls::NlProblem prob;
    try {
        prob = cbls::read_nl(path);
    } catch (const std::exception& e) {
        const std::string what = e.what();
        return (cbls::minlplib::is_reader_coverage_gap(what) ? "reader-unsupported\t"
                                                             : "reader-error\t") +
               one_line(what);
    }
    try {
        cbls::NlToModelOptions opts;
        opts.presolve.enabled = presolve;
        const cbls::NlToModelResult built = cbls::nl_to_model(prob, opts);
        if (built.supported) {
            return presolve ? "modelled" + presolve_cells(built.presolve_stats) : "modelled";
        }
        return "adapter-unsupported\t" + one_line(built.skipped_reasons.empty()
                                                      ? std::string("(no reason given)")
                                                      : built.skipped_reasons[0]);
    } catch (const std::exception& e) {
        return std::string("checker-failed\tmodel building threw: ") + one_line(e.what());
    }
}

}  // namespace

int main(int argc, char** argv) {
    const bool presolve = argc > 1 && std::string(argv[1]) == "--presolve";
    const int first = presolve ? 2 : 1;
    if (argc <= first) {
        std::fprintf(stderr, "usage: %s [--presolve] FILE.nl...\n", argv[0]);
        return 2;
    }
    for (int i = first; i < argc; ++i) {
        const std::string path = argv[i];
        std::printf("%s\t%s\n", one_line(path).c_str(), check(path, presolve).c_str());
        std::fflush(stdout);
    }
    return 0;
}
