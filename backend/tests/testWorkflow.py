from main import ValuationWorkflow

workflow = ValuationWorkflow()
report = workflow.execute("path/to/financials.pdf")
report.save("valuation.docx")