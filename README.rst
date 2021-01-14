Introduction
============
This is a utility library for code shared between `katsdpimager`_ and
`katsdpcontim`_. It is not currently intended to be used directly.

.. _katsdpimager: https://github.com/ska-sa/katsdpimager/
.. _katsdpcontim: https://github.com/ska-sa/katsdpcontim/

Revision history
================

0.1.1
-----
- Workaround for an `Astropy issue`_ that would cause crashes when rendering
  MeerKAT spectral images with Astropy 4.1+.

  .. _Astropy issue: https://github.com/astropy/astropy/issues/11248

- Add LICENSE file and update copyright notices.
- Add ``files`` option in ``mypy.ini`` so that one can just run ``mypy``.

0.1
---
First public release.
