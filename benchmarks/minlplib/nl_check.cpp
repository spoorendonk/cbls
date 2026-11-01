// MINLPLib NL checker: reads and models each NL file named on the command
// line, without solving, and prints one tab-separated line per file:
//
//   <path> \t modelled
//   <path> \t reader-error \t <message>
//   <path> \t adapter-unsupported \t <first reason>
//
// benchmarks/instances/minlplib/status.py runs it over the unpacked MINLPLib
// archive to give every catalogue instance a status (#283). Tabs and newlines
// inside a message are replaced by spaces, so each file is exactly one line.
#include <cbls/io_nl.h>
#include <cstdio>
#include <exception>
#include <string>

namespace {

std::string one_line(std::string s) {
    for (char& c : s) {
        if (c == '\t' || c == '\n' || c == '\r') {
            c = ' ';
        }
    }
    return s;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s FILE.nl...\n", argv[0]);
        return 2;
    }
    for (int i = 1; i < argc; ++i) {
        const std::string path = argv[i];
        std::string status;
        try {
            const cbls::NlProblem prob = cbls::read_nl(path);
            const cbls::NlToModelResult built = cbls::nl_to_model(prob);
            if (built.supported) {
                status = "modelled";
            } else {
                status = "adapter-unsupported\t" + one_line(built.skipped_reasons.empty()
                                                                ? std::string("(no reason given)")
                                                                : built.skipped_reasons[0]);
            }
        } catch (const std::exception& e) {
            status = std::string("reader-error\t") + one_line(e.what());
        }
        std::printf("%s\t%s\n", one_line(path).c_str(), status.c_str());
        std::fflush(stdout);
    }
    return 0;
}
