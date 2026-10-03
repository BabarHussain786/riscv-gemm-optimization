# Fig. 5 status

The workflow is implemented and locally checked, but no K1 performance result
is claimed from this Windows host. The host executable is reference-only and
the plotter rejects empty/non-accepted data. Before using a figure, run the
launcher on K1 and confirm every selected configuration passes validation.

The primary timing is end-to-end wall time. Profile fields are aggregate worker
durations and are explanatory. The comparison is 8-core RVV versus 4 RVV + 4
IME static/dynamic; it is not pure IME versus RVV. The original kernel sources
and legacy Fig. 5 files remain unchanged; the timer-free packing header is
copied verbatim from the repository dispatch except for removed instrumentation.
