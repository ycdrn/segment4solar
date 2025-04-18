# Segment4Solar

Segment4Solar (v0.1) is a tool to analyze facade images for solar applications. 

## Instant BIPV Design using Street View Images

Building-Integrated Photovoltaic (BIPV) retrofits can help transform buildings into energy producers, but accurate, facade-specific solar assessments are essential. Building facades are diverse and complex, with windows, balconies, and other features complicating solar panel installation. Existing approaches to estimating facade solar potential often oversimplify these details, leading to inaccurate overestimates. Segment for Solar Facades (S4S) addresses this by offering a segmentation approach to evaluate solar potential in detail.

S4S combines deep learning-based image segmentation with solar irradiance data to assess the solar potential of building facades. Using the Segment Anything model, S4S performs segmentation on facade images to identify suitable areas for solar panel installation, accounting for windows, doors, and other architectural features. On selected facade surfaces, BIPVs are considered based on user-defined PV technology. S4S then integrates irradiance data from PVGIS to calculate the annual energy potential and monthly energy yields for the facade, considering its unique details.

S4S is currently under development. Future enhancements will further improve energy yield estimation by incorporating shading factors through image analysis and allowing additional user input to refine segmentation results. If you’re interested in learning more or have any ideas, feel free to reach out!

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
