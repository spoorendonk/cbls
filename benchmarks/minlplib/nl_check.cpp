// MINLPLib NL checker: reads and models each NL file named on the command
// line, without solving, and prints one tab-separated line per file:
//
//   <path> \t modelled
//   <path> \t reader-error \t <message>          the file is malformed
//   <path> \t reader-unsupported \t <message>    a construct the reader lacks
//   <path> \t adapter-unsupported \t <reason>    an op the adapter lacks
//   <path> \t checker-failed \t <message>        the adapter or model threw
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

std::string check(const std::string& path) {
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
        const cbls::NlToModelResult built = cbls::nl_to_model(prob);
        if (built.supported) {
            return "modelled";
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
    const int first = 1;
    const auto usage = [&] { std::fprintf(stderr, "usage: %s FILE.nl...\n", argv[0]); };
    if (argc <= first) {
        usage();
        return 2;
    }
    // An option must not become a file name: the census would then report it
    // as a file and exit 0.
    for (int i = first; i < argc; ++i) {
        if (std::string(argv[i]).rfind("--", 0) == 0) {
            std::fprintf(stderr, "misplaced or unknown option: %s\n", argv[i]);
            usage();
            return 2;
        }
    }
    for (int i = first; i < argc; ++i) {
        const std::string path = argv[i];
        std::printf("%s\t%s\n", one_line(path).c_str(), check(path).c_str());
        std::fflush(stdout);
    }
    return 0;
}
