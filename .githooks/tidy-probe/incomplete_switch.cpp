// Gate probe for issue #171 -- deliberately NOT clean, and never built.
//
// .githooks/tidy-probe.sh lints this file under each clang-tidy config
// directory's effective config and FAILS unless all three diagnostics below
// are reported. Each pins one thing the gate needs:
//
//   clang-diagnostic-switch            -- `clang-diagnostic-*` survives the
//                                         `-*` in .clang-tidy's Checks. (Clang
//                                         enables -Wswitch by default, so this
//                                         one needs no flag; GCC, the build
//                                         compiler, needs -Wall for it.)
//   clang-diagnostic-unused-variable   -- the compile command carries -Wall.
//   clang-diagnostic-unused-parameter  -- the compile command carries -Wextra.
//
// The switch is the case that matters: the NodeOp dispatch tables in
// src/dag.cpp and src/io.cpp are written without a `default:` so that an op
// nobody handled is reported, and this is the shape of that report.
//
// It lives under .githooks/ so that neither the build (the CMake target naming
// it is EXCLUDE_FROM_ALL; it exists only to put this file's compile command in
// compile_commands.json) nor the tree sweep in .clang-tidy ever sees it. The
// pre-push hook drops .githooks/ from the files it lints for the same reason.
// Do not "fix" the findings here: the probe passes by being reported.

namespace cbls_tidy_probe {

enum class Probe { A, B, C };

int incomplete_switch(Probe probe, int unused_parameter) {
    int unused_variable = 0;
    switch (probe) {
        case Probe::A:
            return 1;
        case Probe::B:
            return 2;
    }
    return 0;
}

}  // namespace cbls_tidy_probe
