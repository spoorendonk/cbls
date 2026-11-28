```{include} ../README.md
:relative-docs: docs/
:relative-images:
:end-before: <!-- docs/index.md puts the docs site's problem figure here. -->
```

% Drawn by figures/make_routes.py from a cbls solve; one copy per colour mode.

```{figure} figures/routes-light.svg
:figclass: only-light
:align: center
:width: 420px
:alt: Routes fanning out from a depot in the middle of the plane, each a loop through a group of customers and back.

Routes cbls finds for the CVRP instance `X-n115-k10` (Uchoa et al.) from the
depot (the square): every customer, sized by demand, on one route within the
vehicle capacity. A model of `List` variables; no `List` result is published.
```

```{figure} figures/routes-dark.svg
:figclass: only-dark
:align: center
:width: 420px
:alt: Routes fanning out from a depot in the middle of the plane, each a loop through a group of customers and back.

Routes cbls finds for the CVRP instance `X-n115-k10` (Uchoa et al.) from the
depot (the square): every customer, sized by demand, on one route within the
vehicle capacity. A model of `List` variables; no `List` result is published.
```

```{include} ../README.md
:relative-docs: docs/
:relative-images:
:start-after: <!-- docs/index.md puts the docs site's problem figure here. -->
```

```{toctree}
:hidden:

results
cpp
autoapi/cbls/index
examples/minlplib
examples/mip
examples/cvrp
uc-chped
```
