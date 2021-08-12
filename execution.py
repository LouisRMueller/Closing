from Analysis import Analysis

A = Analysis(from_db=True)
# A = Analysis(from_db=False, raw=True)
# A = Analysis(from_db=False, raw=False)


###

print(A.data.head())