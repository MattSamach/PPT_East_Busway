# Pittsburghers For Public Transit - Beyond the East Busway survey results

In this folder:

## `geojson/`  

Map layers used in the survey.

## `Joining Results with Map Layers.ipynb`

A Jupyter notebook showing how to interpret and work with the results as they relate to layers in the `geojson` folder. Includes examples.

## `responses_anonymous_pivotsrc.csv`

Full results table, where rows are responses to individual questions in the survey. 

The `id` field groups a single, complete survey submission.

## `responses_anonymous.csv`

Results table, where rows are complete survey responses and columns are questions. Cells may contain multiple values.

## `question_layer_lookup.csv`

A lookup between questions names in `responses_anonymous_pivotsrc.csv` and the map layer file name for files in the `geojson/` folder.

## `surveyConfig.json`

The raw configuration object used behind the scenes in the survey web app. Includes full survey text and indicates whether questions were required (among other things).
