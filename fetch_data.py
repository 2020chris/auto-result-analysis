import mysql.connector
import pandas as pd
import numpy as np

cnx = mysql.connector.connect(user='qe_automation', password='T3st!ng',
                              host='dv50-mysql-003',
                              database='qe_automation')
cursor = cnx.cursor()

query = "SELECT ti.ExecutionItemID, ts.StepLogID, ti.IssueType, ti.TestCaseStatus, ts.StepDescription, ts.screenshot, ti.ExecutionId \
FROM testexecutionitem ti INNER JOIN teststeplog ts ON ts.ExecutionItemID = ti.ExecutionItemID \
where ti.StartingTime BETWEEN '2023-02-01' AND '2023-03-31' \
AND ts.StepStatus NOT IN ('Passed', 'In Progress', 'Info', 'Failed') \
    AND ti.ExecutionID IN ( \
        SELECT te.ExecutionId \
        FROM testexecution te \
        WHERE te.Region LIKE 'T2%' \
          AND ti.TestCaseStatus != 'Passed' \
          AND ti.AnalysisStatus = 'Analyzed' \
          AND te.ExecutionType IN ('bvt-p2', 'CCIDPreRegression-HQ', 'CCIDRegression-HQ', 'CCIDRegression-IN', 'CCIDRegression-IN-CSN', 'regression') \
        )"
cursor.execute(query)
rows = cursor.fetchall()
cols = [desc[0] for desc in cursor.description]
df = pd.DataFrame(rows, columns=cols)
#df.to_csv('testexecution.csv', index=False)
df.to_parquet('data_feb_march.parquet', compression='brotli')