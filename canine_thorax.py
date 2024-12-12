import openai 
import pandas as pd
from dotenv import load_dotenv
import re
import os

load_dotenv()
# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Path to the Excel file
file_path = 'canine_thorax_scoring.xlsx'

# Load the Excel file
df = pd.read_excel(file_path)

# Fill NaN values with empty strings to avoid concatenation issues
df.fillna('', inplace=True)

# Combine multiple columns for report_1
df['report_1'] = df[['Findings (original radiologist report)', 
                     'Conclusions (original radiologist report)', 
                     'Recommendations (original radiologist report)']].agg(lambda x: '\n\n'.join(x.dropna()), axis=1)

def extract_conditions(radiologist_report_text):
    # Prepare the prompt
    prompt = f"""
{radiologist_report_text}
    """
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": "You are a vet radiologist specialized in radiology report analysis."},
                      {"role": "user", "content": prompt}]
        )
        
        extracted_conditions = response['choices'][0]['message']['content']
        print(f"LLM Response:\n{extracted_conditions}\n")  # Debugging output
        
        # Extract only the explanation part from the response using regex
        explanation_match = re.search(r"\n\n(.*)", extracted_conditions, re.DOTALL)
        explanation_text = explanation_match.group(1).strip() if explanation_match else ""
        
        return extracted_conditions, explanation_text, prompt  # Return response, explanation, and prompt
    
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return None, None, prompt  # Return None if there's an error

# Define conditions to check
conditions_1 = ["perihilar_infiltrate", "pneumonia", "bronchitis", "interstitial", "diseased_lungs", "hypo_plastic_trachea", "cardiomegaly",
                "pulmonary_nodules", "pleural_effusion", "rtm", "focal_caudodorsal_lung", "focal_perihilar", "pulmonary_hypoinflation",
                "right_sided_cardiomegaly", "pericardial_effusion", "bronchiectasis", "pulmonary_vessel_enlargement", "left_sided_cardiomegaly",
                "thoracic_lymphadenopathy", "esophagitis"]

# Initialize a list to hold results
results_list = []

# Iterate through the rows in the original DataFrame and extract classifications
for index, row in df.head(50).iterrows():
    patient_id = row['CaseID']
    report_1 = row['report_1']
    
    print(f"Processing Patient ID: {patient_id}")  # Debugging output

    # Extract classifications and explanation
    classification_text, explanation_text, prompt_text = extract_conditions(report_1)

    # Prepare a new dictionary with CaseID, prompt text, and explanation (llm_proof)
    new_row = {'CaseID': patient_id, 'report_1': prompt_text, 'llm_proof': explanation_text}  # Updated llm_proof column

    # Fill the new_row with defaults first
    for condition in conditions_1:
        new_row[condition.lower().replace(' ', '_')] = "normal"  # Default to Normal

    if classification_text:
        # Use regex to find valid conditions in the classification text
        for condition in conditions_1:
            # Create a regex pattern to find the condition in the output
            pattern = re.compile(rf"{condition.lower()}.*?(normal|abnormal)", re.IGNORECASE)
            match = pattern.search(classification_text)
            if match:
                result = match.group(1).strip().lower()
                new_row[condition.lower().replace(' ', '_')] = result  # Update the result for this condition

    # Append the new_row dictionary to the results list
    results_list.append(new_row)

# Create a DataFrame from the results list
results_df = pd.DataFrame(results_list)

# Load the original data for comparison
df2 = pd.read_excel(file_path)

# Standardize condition column names in results_df: replace underscores with spaces, and convert both to lowercase
results_df.columns = results_df.columns.str.replace(' ', '_').str.lower()

# Ensure df2 also has lowercase column names for consistency
df2.columns = df2.columns.str.lower()

# Merge both dataframes based on 'CaseID' (or the relevant ID column)
merged_df = pd.merge(results_df, df2, on='caseid', suffixes=('_original_radiologist_report', '_ai_report'))
print("Merged DataFrame:")
print(merged_df.head())

# Create empty columns for True Positive, True Negative, False Positive, and False Negative
merged_df['True Positive'] = ""
merged_df['True Negative'] = ""
merged_df['False Positive'] = ""
merged_df['False Negative'] = ""

# List of condition columns (standardized to match space-separated names)
conditions = ["perihilar_infiltrate", "pneumonia", "bronchitis", "interstitial", "diseased_lungs", "hypo_plastic_trachea", "cardiomegaly",
                "pulmonary_nodules", "pleural_effusion", "rtm", "focal_caudodorsal_lung", "focal_perihilar", "pulmonary_hypoinflation",
                "right_sided_cardiomegaly", "pericardial_effusion", "bronchiectasis", "pulmonary_vessel_enlargement", "left_sided_cardiomegaly",
                "thoracic_lymphadenopathy", "esophagitis"]

# Function to compare conditions and categorize results
def compare_conditions(row, condition):
    # Ensure the values are strings before calling .lower()
    cond_file1 = str(row[f'{condition}_original_radiologist_report']).lower()
    cond_file2 = str(row[f'{condition}_ai_report']).lower()
    
    # True Positive: Both are Abnormal
    if cond_file1 == 'abnormal' and cond_file2 == 'abnormal':
        return 'tp', condition
    # True Negative: Both are Normal
    elif cond_file1 == 'normal' and cond_file2 == 'normal':
        return 'tn', condition
    # False Positive: File 2 is Abnormal, File 1 is Normal
    elif cond_file1 == 'normal' and cond_file2 == 'abnormal':
        return 'fp', condition
    # False Negative: File 2 is Normal, File 1 is Abnormal
    elif cond_file1 == 'abnormal' and cond_file2 == 'normal':
        return 'fn', condition
    else:
        return None, None

# Iterate through each row and each condition, and append the condition to the relevant result column
for index, row in merged_df.iterrows():
    for condition in conditions:
        result, condition_name = compare_conditions(row, condition)
        
        # Append the condition name to the corresponding column
        if result == 'tp':
            merged_df.at[index, 'True Positive'] += condition_name + ', '
        elif result == 'tn':
            merged_df.at[index, 'True Negative'] += condition_name + ', '
        elif result == 'fp':
            merged_df.at[index, 'False Positive'] += condition_name + ', '
        elif result == 'fn':
            merged_df.at[index, 'False Negative'] += condition_name + ', '

# Clean up: Remove any trailing commas and whitespace from the result columns
merged_df['True Positive'] = merged_df['True Positive'].str.rstrip(', ')
merged_df['True Negative'] = merged_df['True Negative'].str.rstrip(', ')
merged_df['False Positive'] = merged_df['False Positive'].str.rstrip(', ')
merged_df['False Negative'] = merged_df['False Negative'].str.rstrip(', ')

# Save the merged DataFrame to an Excel file
output_file_path = 'final_conditions_cleaned_canine_thorax.xlsx'
merged_df.to_excel(output_file_path, index=False, engine='openpyxl')

print(f'Final DataFrame saved to {output_file_path}')

# Display the final DataFrame with the updated results
final_df = merged_df[['caseid', 'True Positive', 'True Negative', 'False Positive', 'False Negative', 'report_1', 'llm_proof']]
print(final_df)
