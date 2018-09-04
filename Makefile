# This Makefile is for convenience as a reminder and shortcut for the most used commands

# Package folder
PACKAGE = kerrgeodesic_gw

# Change to your sage command if needed
SAGE = sage

all: install test

install:
	$(SAGE) -pip install --upgrade --no-index -v .

uninstall:
	$(SAGE) -pip uninstall .

develop:
	$(SAGE) -pip install --upgrade --no-index -v -e .

test:
	$(SAGE) setup.py test

coverage:
	$(SAGE) -coverage $(PACKAGE)/*

doc:
	cd docs && $(SAGE) -sh -c "make html"

doc-pdf:
	cd docs && $(SAGE) -sh -c "make latexpdf"

clean: clean-doc

clean-doc:
	cd docs && $(SAGE) -sh -c "make clean"

.PHONY: all install develop test coverage clean clean-doc doc doc-pdf
