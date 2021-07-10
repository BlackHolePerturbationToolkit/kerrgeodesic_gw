FROM sagemath/sagemath:9.3.beta8

RUN sage -pip install kerrgeodesic_gw
#RUN echo "jupyter-nbextension install rise --py --sys-prefix" | sage -sh
#RUN echo "jupyter-nbextension enable rise --py --sys-prefix" | sage -sh

# Inspired from https://mybinder.readthedocs.io/en/latest/dockerfile.html#preparing-your-dockerfile
# Make sure the contents of our repo are in ${HOME}
COPY . ${HOME}
