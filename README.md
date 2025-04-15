# Segment4Solar

Segment4Solar (v0.1) is a project focusing on the analysis and segmentation of facade imagery for solar applications. 

Segment4Solar leverages two APIs to:
- **Detect available areas on facades,** where solar panels can potentially be applied onto.
- **Estimate monthly and annual energy generation** to understand how much of your demand can be met by the solar panels.

By transforming raw facade imagery into actionable data, Segment4Solar aids in the planning and management of solar installations in urban environments.

## Demo

![Segment4Solar Demo](demo.gif)

## Reference

For a detailed explanation of the improvements in the estimated energy yield with the methodologies used in our analysis tool, please refer to our [[conference paper]](https://iopscience.iop.org/article/10.1088/1742-6596/2600/4/042005/pdf)

```bibtex
@article{Duran_2023,
doi = {10.1088/1742-6596/2600/4/042005},
url = {https://dx.doi.org/10.1088/1742-6596/2600/4/042005},
year = {2023},
month = {nov},
publisher = {IOP Publishing},
volume = {2600},
number = {4},
pages = {042005},
author = {Duran, Ayça and Waibel, Christoph and Schlueter, Arno},
title = {Estimating surface utilization factors for BIPV applications using pix2pix on street captured façade images},
journal = {Journal of Physics: Conference Series},
}
```

## Acknowledgements

- [Segment Anything Model](https://arxiv.org/abs/2304.02643)
- [PVGIS](https://re.jrc.ec.europa.eu/pvg_tools/en/)
    
This research was conducted at the Future Cities Lab Global at ETH Zurich. Future Cities Lab Global is supported and funded by the National Research Foundation, Prime Minister’s Office, Singapore under its Campus for Research Excellence and Technological Enterprise (CREATE) programme and ETH Zurich (ETHZ), with additional contributions from the National University of Singapore (NUS), Nanyang Technological University (NTU), Singapore and the Singapore University of Technology and Design (SUTD). This study is supported by the A/T doctoral fellowship offered by the Institute of Technology in Architecture at ETH Zurich.
