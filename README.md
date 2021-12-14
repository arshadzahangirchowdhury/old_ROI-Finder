# AI_XRF
Software for X-ray fluorescence imaging analysis using AI tools.

## Instructions

1. Install the packages via the AI_XRF_env.yml.
2. The GUI based example workflows contains a segmenter and an annotator tool.
3. Segmenter tool is used to select .h5 files containing XRF images and extract region of interests (cells).
4. Annotator tool is used to bin the extracted cells in two groups, accepts or rejects corresponding to alive or dead cells.

## Segmenter workflow

The segmenter workflow is designed to identify and explore the parameters which affect the conversion process of images to binary images.

## Annotator workflow

The annotator tool allows the user to bin the XRF images into two categories called "accepts" and "rejects". These two categories can correspond to "live" and "dead" bacterial cells respectively. The user can preview the extracted cells from the segmenter workflow and then use the buttons to annotate and bin the data into two groups. The annotated data is stored to the user's local hard drive inside the annotated_XRF folder. This directory must not be renamed or mvoed.

## Optional parameters 

1. In example workflows, change definitions inside config.py to adjust how figures are rendered.
