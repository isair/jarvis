# Console output

`configure_utf8_streams` prepares an explicit stream owner with `stdout` and
`stderr` attributes. Available text writers with `reconfigure` use UTF-8 and
replacement error handling in place. Open streams and their binary buffers
retain their existing ownership; preparation does not create another wrapper.
Writers without `reconfigure` remain usable through their own text interface.
Missing or closed streams receive Unicode-capable null-device sinks that retain
no output in memory or on disk.

An encoding reconfiguration failure on one stream is logged and does not prevent
preparation of the other. Desktop smoke startup uses this helper on every OS;
Windows source desktop and daemon startup use the same helper. Debug logs record
setup and encoding failures without captured output.
