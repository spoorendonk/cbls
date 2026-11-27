#!/usr/bin/env bash
# Builds the pinned SCIP for cbls_scip_traj (#442): scipopt/scip at SCIP_COMMIT
# beside this file, SoPlex, Ipopt over MUMPS with METIS 5.2.1, TPI=tny.
#
#   benchmarks/common/scip_driver/build_scip.sh PREFIX    # sources in PREFIX/src
#
# Follows scipoptsuite-deploy main e4f0441 (.github/workflows/scripts/linux.bash)
# except: shared libraries, the system's LAPACK/BLAS, no GCG, no Boost. Run it
# niced and pinned to the cores the machine's rules allow; JOBS caps make.
set -euo pipefail

[ $# -eq 1 ] || { echo "usage: $0 PREFIX" >&2; exit 2; }
HERE=$(cd "$(dirname "$0")" && pwd)
SCIP_COMMIT=$(head -n1 "$HERE/SCIP_COMMIT")
SOPLEX_VERSION=8.1.0
IPOPT_VERSION=3.14.19
METIS_VERSION=5.2.1
GKLIB_COMMIT=8bd6bad750b2b0d90800c632cf18e8ee93ad72d7    # deploy main's pin; GKlib has no 5.2.1 tag
MUMPS_TP_COMMIT=35e03ac76c20c2a8b70c0e6ad3604455590518d6 # coin-or-tools/ThirdParty-Mumps HEAD, 2026-11-27
JOBS=${JOBS:-4}

mkdir -p "$1"
PREFIX=$(cd "$1" && pwd)
SRC=$PREFIX/src
mkdir -p "$SRC" "$PREFIX/share"
cd "$SRC"

# Autoconf (MUMPS, Ipopt) reads $prefix/share/config.site: METIS 5.2.1 and LAPACK.
cat > "$PREFIX/share/config.site" <<EOF
enable_java=no
enable_sipopt=no
with_pic=yes
with_metis_cflags="-I$PREFIX/include"
with_metis_lflags="-L$PREFIX/lib -lmetis -lGKlib -lm"
with_lapack_lflags="-llapack -lblas -lgfortran -lm"
EOF

if [ ! -f "$PREFIX/lib/libmetis.a" ]; then
  [ -d GKlib-$GKLIB_COMMIT ] || { wget -q https://github.com/KarypisLab/GKlib/archive/$GKLIB_COMMIT.tar.gz; tar -xf $GKLIB_COMMIT.tar.gz; }
  [ -d METIS-$METIS_VERSION ] || { wget -q https://github.com/KarypisLab/METIS/archive/refs/tags/v$METIS_VERSION.tar.gz; tar -xf v$METIS_VERSION.tar.gz; }
  (cd GKlib-$GKLIB_COMMIT
   sed -i 's/^CONFIG_FLAGS =/CONFIG_FLAGS = -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_POSITION_INDEPENDENT_CODE=ON/' Makefile
   sed -i 's/-march=native//' GKlibSystem.cmake
   make config prefix="$PREFIX" && make -j"$JOBS" && make install)
  (cd METIS-$METIS_VERSION
   sed -i 's/^CONFIG_FLAGS =/CONFIG_FLAGS = -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_POSITION_INDEPENDENT_CODE=ON/' Makefile
   sed -i 's/-march=native//' conf/gkbuild.cmake
   make config prefix="$PREFIX" gklib_path="$PREFIX" && make -j"$JOBS" && make install)
fi

if [ ! -f "$PREFIX/lib/pkgconfig/coinmumps.pc" ]; then
  [ -d ThirdParty-Mumps ] || git clone -q https://github.com/coin-or-tools/ThirdParty-Mumps.git
  (cd ThirdParty-Mumps && git checkout -q $MUMPS_TP_COMMIT
   [ -d MUMPS ] || ./get.Mumps
   ./configure --prefix="$PREFIX"
   make -j1 && make install)   # serial: MUMPS's Fortran module dependencies race under parallel make
fi

if [ ! -f "$PREFIX/lib/pkgconfig/ipopt.pc" ]; then
  [ -d Ipopt-releases-$IPOPT_VERSION ] || { wget -q https://github.com/coin-or/Ipopt/archive/refs/tags/releases/$IPOPT_VERSION.tar.gz -O ipopt-$IPOPT_VERSION.tar.gz; tar -xf ipopt-$IPOPT_VERSION.tar.gz; }
  mkdir -p Ipopt-releases-$IPOPT_VERSION/build
  (cd Ipopt-releases-$IPOPT_VERSION/build && ../configure --prefix="$PREFIX" && make -j"$JOBS" && make install)
fi

if [ ! -d "$PREFIX/lib/cmake/soplex" ]; then
  [ -d soplex-$SOPLEX_VERSION ] || { wget -q https://github.com/scipopt/soplex/archive/refs/tags/v$SOPLEX_VERSION.tar.gz -O soplex-$SOPLEX_VERSION.tar.gz; tar -xf soplex-$SOPLEX_VERSION.tar.gz; }
  cmake -S soplex-$SOPLEX_VERSION -B soplex-$SOPLEX_VERSION/build -DCMAKE_INSTALL_PREFIX="$PREFIX" \
    -DCMAKE_BUILD_TYPE=Release -DGMP=false -DPAPILO=false -DBOOST=false -DMPFR=false
  cmake --build soplex-$SOPLEX_VERSION/build -j"$JOBS"
  cmake --install soplex-$SOPLEX_VERSION/build
fi

[ -d scip ] || git clone -q --filter=blob:none https://github.com/scipopt/scip.git
git -C scip fetch -q origin "$SCIP_COMMIT" 2>/dev/null || true
git -C scip checkout -q "$SCIP_COMMIT"
cmake -S scip -B scip/build --preset interface -DCMAKE_INSTALL_PREFIX="$PREFIX" \
  -DCMAKE_BUILD_TYPE=Release -DSHARED=true -DLPS=spx -DSYM=snauty -DSOPLEX_DIR="$PREFIX" \
  -DPAPILO=false -DZIMPL=false -DGMP=false -DREADLINE=false -DIPOPT=true -DIPOPT_DIR="$PREFIX" \
  -DTPI=tny -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build scip/build -j"$JOBS"
cmake --install scip/build
"$PREFIX/bin/scip" -c quit | head -n1
