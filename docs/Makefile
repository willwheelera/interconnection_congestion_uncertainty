# Minimal Sphinx Makefile
SPHINXBUILD   = uv run --group docs sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

.PHONY: html clean

html:
	$(SPHINXBUILD) -b html $(SOURCEDIR) $(BUILDDIR)/html

clean:
	rm -rf $(BUILDDIR)
