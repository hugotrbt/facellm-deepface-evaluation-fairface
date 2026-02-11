Analyze the facial image and provide your best visual estimates for the attributes below.

Base your answers only on visible facial characteristics.
You are allowed to make reasonable visual assumptions.
If you are uncertain, still provide your best estimate and reflect uncertainty using a low confidence score.

Return the result as a strictly valid JSON object with exactly the following structure and no additional text:

{
"gender": "",
"ethnicity": "",
"age_range": "",
"confidence": {
"gender": 0.0,
"ethnicity": 0.0,
"age": 0.0
}
}

Allowed values:

gender: Male | Female
ethnicity: Asian | Black | White | Latino_Hispanic | Middle Eastern | Indian
age_range: 0-2 | 3-9 | 10-19 | 20-29 | 30-39 | 40-49 | 50-59 | 60-69 | 70+

Choose exactly one value per attribute.
