import pandas as pd

# Open the teams csv file
terms_df = pd.read_csv("search_terms.csv", index_col=False)
# Get the list of main team names
main_team_names = list(terms_df["term"].astype("string"))
# Get the list of alternate team names
alt_team_names = list(terms_df["alt_term"].astype("string"))
# Combine them
all_terms = main_team_names + alt_team_names
# Get the list of (main_name, alt_name) tuples
team_tuples = list(zip(main_team_names, alt_team_names))
