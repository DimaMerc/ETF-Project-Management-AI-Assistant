import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import docx2txt
import PyPDF2
import io
import base64
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from datetime import datetime, timedelta

# Assume LLMAssistedProjectEstimator is imported and initialized as before
from etf_project_management_ai import LLMAssistedProjectEstimator

# Initialize the AI
@st.cache_resource
def load_ai():
    return LLMAssistedProjectEstimator()

pm_ai = load_ai()

# Initialize session state to store estimation results
if 'estimation_results' not in st.session_state:
    st.session_state.estimation_results = None

st.title('ETF Project Management AI Assistant')

uploaded_file = st.file_uploader("Upload project scope document", type=["txt", "docx", "pdf"])

if uploaded_file is not None:
    # Read and process the file
    if uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        document_content = docx2txt.process(uploaded_file)
    elif uploaded_file.type == "application/pdf":
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
        document_content = "\n".join(page.extract_text() for page in pdf_reader.pages)
    else:
        document_content = uploaded_file.read().decode('utf-8')

    # Extract information
    extracted_info = pm_ai.extract_initial_info(document_content)

    # Display extracted information
    st.header("Extracted Project Information")
    for key, value in extracted_info.items():
        if key != 'additional_variables':
            st.write(f"{key.replace('_', ' ').title()}: {value}")

    if extracted_info['additional_variables']:
        st.subheader("Additional Variables Identified")
        for var in extracted_info['additional_variables']:
            st.write(f"- {var}")
    else:
        st.subheader("No Additional Variables Identified")

    # Input fields for estimation
    st.header("Project Estimation")
    st.write("Enter or modify values for estimation (leave blank if unknown):")
    estimation_values = {}
    for key, value in extracted_info.items():
        if key != 'additional_variables':
            user_input = st.text_input(f"Value for {key.replace('_', ' ').title()}", value=value if value != "Not specified" else "")
            if user_input and user_input != "Not specified":
                estimation_values[key] = user_input

    if st.button("Estimate Project") or st.session_state.estimation_results is not None:
        if not estimation_values:
            st.warning("Please enter values for at least one variable before estimating.")
        else:
            # Perform estimation if not already done
            if st.session_state.estimation_results is None:
                st.session_state.estimation_results = pm_ai.estimate_project(list(estimation_values.keys()), estimation_values)
            
            estimation = st.session_state.estimation_results
            
            st.subheader("Estimation Results")
            if "error" in estimation:
                st.error(f"Error in project estimation: {estimation['error']}")
            else:
                # Complexity visualization
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = estimation['complexity']['score'],
                    title = {'text': "Complexity Score"},
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    gauge = {
                        'axis': {'range': [0, 10]},
                        'steps': [
                            {'range': [0, 3.33], 'color': "lightgreen", 'name': 'Low'},
                            {'range': [3.33, 6.66], 'color': "yellow", 'name': 'Medium'},
                            {'range': [6.66, 10], 'color': "red", 'name': 'High'}
                        ],
                        'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 9}
                    }
                ))
                fig.update_layout(
                    annotations=[dict(x=0.5, y=-0.1, showarrow=False, text="Low: 0-3.33 | Medium: 3.33-6.66 | High: 6.66-10", xref="paper", yref="paper")]
                )
                st.plotly_chart(fig)

                st.write(f"Complexity Explanation: {estimation['complexity']['explanation']}")

                # Duration handling
                duration_estimate = estimation['duration']['estimate']
                duration_explanation = estimation['duration']['explanation']
                try:
                    base_duration = float(duration_estimate.split()[0])
                    st.write(f"Duration Estimate: {duration_estimate}")
                    st.write(f"Duration Explanation: {duration_explanation}")
                except (ValueError, IndexError):
                    st.warning(f"Unable to parse duration: {duration_estimate}. Using default value of 180 days.")
                    base_duration = 180.0

                # Cost handling
                cost_estimate = estimation['cost']['estimate']
                cost_explanation = estimation['cost']['explanation']
                parsed_cost = estimation['cost'].get('parsed_estimate')

                st.write(f"Cost Estimate: {cost_estimate}")
                st.write(f"Cost Explanation: {cost_explanation}")

                try:
                    # Try to convert parsed_cost to float if it's a string
                    if isinstance(parsed_cost, str):
                        parsed_cost = float(parsed_cost.replace(',', ''))
    
                    # Now format the float value
                    st.write(f"Parsed Cost (for calculations): ${parsed_cost:,.2f}")
                except (ValueError, TypeError):
                    st.error(f"Unable to parse cost value: {parsed_cost}. Using 0 for calculations.")
                    st.write("Parsed Cost (for calculations): $0.00")
                    parsed_cost = 0.0

                # Store the parsed_cost as a float for further calculations
                st.session_state['parsed_cost'] = float(parsed_cost)


               
                # Risk Analysis
                st.subheader("Risk Analysis")
                risk_matrix = go.Figure(data=go.Heatmap(
                    z=[[1, 2, 3], [2, 4, 6], [3, 6, 9]],
                    x=['Low', 'Medium', 'High'],
                    y=['Low', 'Medium', 'High'],
                    colorscale='RdYlGn_r'
                ))
                risk_matrix.update_layout(title='Risk Matrix', xaxis_title='Impact', yaxis_title='Likelihood')
                st.plotly_chart(risk_matrix)

                # Team Composition
                st.subheader("Recommended Team Composition")
                team_comp = {
                    'Project Manager': 0.2,
                    'ETF Specialist': 0.3,
                    'Data Engineer': 0.15,
                    'Financial Analyst': 0.15,
                    'Compliance Officer': 0.1,
                    'Software Developer': 0.1
                }
                fig = go.Figure(data=[go.Pie(labels=list(team_comp.keys()), values=list(team_comp.values()))])
                st.plotly_chart(fig)

                # Project Timeline
                st.subheader("Project Timeline")
                start_date_str = estimation_values.get('start_date', '2024-01-01')
                try:
                    start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
                except ValueError:
                    st.error(f"Unable to parse start date: {start_date_str}. Using default date 2024-01-01.")
                    start_date = datetime(2024, 1, 1)

                end_date = start_date + timedelta(days=int(base_duration))
            
                tasks = [
                    dict(Task="Planning", Start=start_date, Finish=start_date + timedelta(days=base_duration*0.2)),
                    dict(Task="Development", Start=start_date + timedelta(days=base_duration*0.2), Finish=start_date + timedelta(days=base_duration*0.7)),
                    dict(Task="Testing", Start=start_date + timedelta(days=base_duration*0.7), Finish=start_date + timedelta(days=base_duration*0.9)),
                    dict(Task="Deployment", Start=start_date + timedelta(days=base_duration*0.9), Finish=end_date)
                ]
                
                fig = go.Figure([go.Bar(
                    base=[(task['Start'] - start_date).days for task in tasks],
                    x=[(task['Finish'] - task['Start']).days for task in tasks],
                    y=[task['Task'] for task in tasks],
                    orientation='h'
                )])
                fig.update_layout(title='Project Timeline', xaxis_title='Days', yaxis_title='Task', height=400)
                st.plotly_chart(fig)

                # Sensitivity Analysis
                st.subheader("Project Sensitivity Analysis")
                factors = ['Resource Availability', 'Technology Changes', 'Regulatory Complexity', 'Integration Challenges']
                impact = [0.3, 0.2, 0.3, 0.2]
                fig = go.Figure(data=[go.Bar(x=factors, y=impact)])
                fig.update_layout(title='Impact of Various Factors on Project', xaxis_title='Factors', yaxis_title='Impact Score')
                st.plotly_chart(fig)

                # Export Functionality
                st.subheader("Export Analysis")
                if st.button("Export as PDF"):
                    pdf_buffer = io.BytesIO()
                    c = canvas.Canvas(pdf_buffer, pagesize=letter)
                    c.drawString(100, 750, "ETF Project Analysis Report")
                    c.drawString(100, 700, f"Complexity Score: {estimation['complexity']['score']}")
                    c.drawString(100, 650, f"Duration Estimate: {estimation['duration']['estimate']}")
                    c.drawString(100, 600, f"Cost Estimate: {estimation['cost']['estimate']}")
                    c.save()
                    pdf_buffer.seek(0)
                    b64 = base64.b64encode(pdf_buffer.getvalue()).decode()
                    href = f'<a href="data:application/octet-stream;base64,{b64}" download="etf_project_analysis.pdf">Download PDF Report</a>'
                    st.markdown(href, unsafe_allow_html=True)

                # Q&A Interface
                st.subheader("Ask Follow-up Questions")
                user_question = st.text_input("Enter your question about the project:")
                if user_question:
                    # Simulated AI response - replace with actual AI processing later
                    st.write(f"AI Response: Based on the analysis, regarding '{user_question}', we recommend...")

                # Interactive Simulation
                st.subheader("Interactive Project Scenario Analysis")
                resource_availability = st.slider("Adjust resource availability:", -1.0, 1.0, 0.0)
                tech_complexity = st.slider("Adjust technological complexity:", -1.0, 1.0, 0.0)
                regulatory_change = st.checkbox("Include potential regulatory changes")

                # Calculate impact
                impact = (resource_availability * 0.3 + tech_complexity * 0.2 + (0.3 if regulatory_change else 0)) / 0.8
                
                # Recalculate duration and cost based on impact
                adjusted_duration = base_duration * (1 + impact)
                adjusted_cost = parsed_cost * (1 + impact)
                
                st.write(f"Adjusted Duration: {adjusted_duration:.0f} days")
                st.write(f"Adjusted Cost: ${adjusted_cost:,.2f}")

                # Update the Project Timeline based on adjusted duration
                st.subheader("Updated Project Timeline")
                end_date = start_date + timedelta(days=int(adjusted_duration))
                
                tasks = [
                    dict(Task="Planning", Start=start_date, Finish=start_date + timedelta(days=adjusted_duration*0.2)),
                    dict(Task="Development", Start=start_date + timedelta(days=adjusted_duration*0.2), Finish=start_date + timedelta(days=adjusted_duration*0.7)),
                    dict(Task="Testing", Start=start_date + timedelta(days=adjusted_duration*0.7), Finish=start_date + timedelta(days=adjusted_duration*0.9)),
                    dict(Task="Deployment", Start=start_date + timedelta(days=adjusted_duration*0.9), Finish=end_date)
                ]
                
                fig = go.Figure([go.Bar(
                    base=[(task['Start'] - start_date).days for task in tasks],
                    x=[(task['Finish'] - task['Start']).days for task in tasks],
                    y=[task['Task'] for task in tasks],
                    orientation='h'
                )])
                fig.update_layout(title='Updated Project Timeline', xaxis_title='Days', yaxis_title='Task', height=400)
                st.plotly_chart(fig)

    else:
        st.info("Please enter project details and click 'Estimate Project' to see the analysis.")

else:
    st.info("Please upload a project scope document to begin analysis.")