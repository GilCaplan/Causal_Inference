"""Download full folktables ACS PUMS dataset: all states, 2018 1-year."""
from folktables import ACSDataSource

STATES = [
    "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","ID","IL","IN","IA",
    "KS","KY","LA","ME","MD","MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ",
    "NM","NY","NC","ND","OH","OK","OR","PA","RI","SC","SD","TN","TX","UT","VT",
    "VA","WA","WV","WI","WY","PR",
]

src = ACSDataSource(survey_year="2018", horizon="1-Year", survey="person",
                    root_dir="data")
data = src.get_data(states=STATES, download=True)
print(f"Rows: {len(data):,}  Cols: {len(data.columns)}")
