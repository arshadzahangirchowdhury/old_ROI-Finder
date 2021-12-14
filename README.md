# ROI-Finder
Software for X-ray fluorescence imaging analysis using AI tools. The goal of development is to find regions of interest in X-ray fluoroscence images to better interpret the collected XRF data at the Advanced Photon Source. Based on the regions of interest, experiments will be steered to take finer resolution scans.

## Installation

1. git clone https://github.com/arshadzahangirchowdhury/ROI-Finder
2. Install the packages via the AI_XRF_env.yml file.

## Instructions

1. The GUI based example workflows contains a segmenter, an annotator and a recommender tool.
2. Segmenter tool is used to select .h5 files containing XRF images and extract region of interests (cells).
3. Annotator tool is used to bin the extracted cells in two groups, accepts or rejects corresponding to alive or dead cells.

## Segmenter workflow

The segmenter workflow is designed to identify and explore the parameters which affect the conversion process of images to binary images.

## Annotator workflow

The annotator tool allows the user to bin the XRF images into two categories called "accepts" and "rejects". These two categories can correspond to "live" and "dead" bacterial cells respectively. The user can preview the extracted cells from the segmenter workflow and then use the buttons to annotate and bin the data into two groups. The annotated data is stored to the user's local hard drive inside the annotated_XRF folder. This directory must not be renamed or moved.

## Recommender workflow

The recommender tool allows the user to select an AI method, based on which recommendations are given to the user based on bacterial cells which are similar to selected cells by the user.
 

## How to set up the preferences for the workflows? 

In the example workflows directory, change the definitions inside config.py to adjust how figure, axes, colorbars and labels are rendered.

## Who is the user?
The software is being developed for Biologists and X-ray physicists.

## License

BSD License is pending approval.
