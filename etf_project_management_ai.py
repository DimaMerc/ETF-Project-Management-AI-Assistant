import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
import yfinance as yf
from openai import OpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
import json
from json.decoder import JSONDecodeError
import os
import sqlite3
from typing import List, Dict, Any
import logging
from datetime import datetime, timedelta
from dateutil import parser
import re
from openai import OpenAIError
from env import OPENAI_API_KEY

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class LLMAssistedProjectEstimator:
    def __init__(self, model_name: str = "gpt-4o", db_path: str = "project_memory.db"):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model_name = model_name
        self.db_path = db_path
        self.init_db()
        self.common_variables: List[str] = [
            "project_name", "start_date", "end_date", "duration", "budget", 
            "team_size", "stakeholders", "resource_costs", "etf_type", "underlying_assets",
            "regulatory_requirements", "technology_stack", "risk_assessment", "market_volatility",
            "data_sources", "trading_frequency", "historical_performance", "competitor_analysis"
        ]

    def init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS projects (
                id INTEGER PRIMARY KEY,
                project_data TEXT,
                estimation_data TEXT
            )
        ''')
        conn.commit()
        conn.close()

    def save_project(self, project_data: Dict[str, Any], estimation_data: Dict[str, Any]):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO projects (project_data, estimation_data) VALUES (?, ?)",
            (json.dumps(project_data), json.dumps(estimation_data))
        )
        conn.commit()
        conn.close()

    def get_past_projects(self, limit: int = 5) -> List[Dict[str, Any]]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT project_data, estimation_data FROM projects ORDER BY id DESC LIMIT ?", (limit,))
        projects = [
            {
                "project_data": json.loads(row[0]),
                "estimation_data": json.loads(row[1])
            }
            for row in cursor.fetchall()
        ]
        conn.close()
        return projects
    
    def parse_date(self, date_string):
        date_formats = ['%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y']
        for fmt in date_formats:
            try:
                return datetime.strptime(date_string, fmt)
            except ValueError:
                continue
        print(f"Warning: Unable to parse date: {date_string}. Returning None.")
        return None

    def extract_initial_info(self, document: str) -> Dict[str, Any]:
        messages = [
           {"role": "system", "content": "You are an AI assistant specialized in extracting project information for ETF (Exchange-Traded Fund) projects."},
           {"role": "user", "content": f"""
           Given the following project document, please extract the requested information and provide it in the specified JSON format:

           Document:
           {document}

           Please provide your response in the following JSON format, filling in the values based on the document content:

           {{
              "project_name": "Name of the ETF project",
              "start_date": "Project start date if specified, otherwise 'Not specified'",
              "end_date": "Project end date if specified, otherwise 'Not specified'",
              "duration": "Project duration if specified, otherwise 'Not specified'",
              "budget": "Project budget if specified, otherwise 'Not specified'",
              "team_size": "Number of team members if specified, otherwise 'Not specified'",
              "stakeholders": "Number of stakeholders if specified, otherwise 'Not specified'",
              "resource_costs": "Summary of resource costs if specified, otherwise 'Not specified'",
              "etf_type": "Type of ETF if specified, otherwise 'Not specified'",
              "underlying_assets": "Description of underlying assets if specified, otherwise 'Not specified'",
              "regulatory_requirements": "Any mentioned regulatory requirements, otherwise 'Not specified'",
              "technology_stack": "Mentioned technology stack or systems, otherwise 'Not specified'",
              "risk_assessment": "Any mentioned risk factors or assessments, otherwise 'Not specified'",
              "market_volatility": "Any mention of market volatility considerations, otherwise 'Not specified'",
              "data_sources": "Mentioned data sources or providers, otherwise 'Not specified'",
              "trading_frequency": "Mentioned trading frequency or rebalancing schedule, otherwise 'Not specified'",
              "historical_performance": "Any mention of historical performance analysis, otherwise 'Not specified'",
              "competitor_analysis": "Any mention of competitor analysis, otherwise 'Not specified'",
              "additional_variables": []
            }}

            Ensure your response is a valid JSON string without any additional text or formatting.
            If a piece of information is not found in the document, use "Not specified" as the value.
            Pay special attention to implicit information and make reasonable inferences where possible.
            """
            }
        ]

       # logger.info(f"Attempting to use model: {self.model_name}")
    
        try:
           # logger.info("Initiating API call")
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages
            )
          #  logger.info("API call completed successfully")
           # logger.info(f"Raw API response: {response}")
        
            result = response.choices[0].message.content
           # logger.info(f"Extracted content: {result}")
        

         # Remove markdown formatting if present
            result = result.strip('`').strip()
            if result.startswith('json'):
                result = result[4:].strip()


            extracted_info = json.loads(result)
           # logger.info("Successfully parsed JSON response")
        
        # Post-processing of extracted information
            self._process_team_size(extracted_info)
            self._process_duration(extracted_info)
            extracted_info = self._process_dates(extracted_info)
        
            return extracted_info
    
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON: {e}")
            logger.error(f"Raw content causing JSON error: {result}")
            return self._get_default_info()
    
        except OpenAIError as e:
            logger.error(f"OpenAI API error: {type(e).__name__} - {e}")
            return self._get_default_info()
    
        except Exception as e:
            logger.error(f"Unexpected error during API call: {type(e).__name__} - {e}")
            return self._get_default_info()

    def _process_team_size(self, info: Dict[str, Any]):
        if info['team_size'] == "Not specified":
            resource_costs = info.get('resource_costs', {})
            if isinstance(resource_costs, dict):
                info['team_size'] = len(resource_costs)

    def _process_duration(self, info: Dict[str, Any]):
        if info['duration'] != "Not specified":
            try:
                if '-' in info['duration']:
                    min_duration, max_duration = map(int, info['duration'].split('-'))
                    info['duration'] = f"{min_duration}-{max_duration} months"
                else:
                    duration = int(info['duration'])
                    info['duration'] = f"{duration} months"
            except ValueError:
                logger.warning(f"Unable to parse duration: {info['duration']}. Keeping original value.")

    def _process_dates(self, info: Dict[str, Any]):
        if info['start_date'] != "Not specified" and info['end_date'] != "Not specified":
            try:
                start_date = self.parse_date(info['start_date'])
                end_date = self.parse_date(info['end_date'])
                if start_date and end_date:
                    duration = (end_date - start_date).days
                    info['duration'] = f"{duration} days"
                    info['start_date'] = start_date.strftime('%Y-%m-%d')
                    info['end_date'] = end_date.strftime('%Y-%m-%d')
                    print(f"Successfully processed dates: Start {info['start_date']}, End {info['end_date']}, Duration {info['duration']}")
                else:
                    raise ValueError("Unable to parse one or both dates")
            except ValueError as e:
                print(f"Date processing error: {e}. Using original date strings.")
                info['duration'] = "Unable to calculate"
        else:
            print("Start date or end date not specified.")
            info['duration'] = "Not specified"
        return info

    def _get_default_info(self) -> Dict[str, Any]:
        return {var: "Not specified" for var in self.common_variables + ["additional_variables"]}
    
    
    def parse_cost(self, cost_string):
        try:
        # Remove any non-alphanumeric characters except '.', '-', and spaces
            cost_string = ''.join(c for c in cost_string if c.isalnum() or c in '.- ')
            cost_string = cost_string.lower().replace('million', '').strip()
        
            if '-' in cost_string:
                parts = cost_string.split('-')
                costs = [float(part.strip()) for part in parts]
                avg_cost = sum(costs) / len(costs)
            else:
                avg_cost = float(cost_string)
        
        # If the original string contained 'million', multiply by 1,000,000
            if 'million' in cost_string.lower():
                avg_cost *= 1_000_000
        
            return avg_cost
        except ValueError:
            print(f"Warning: Unable to parse cost: {cost_string}. Returning 0.")
            return 0.0

    def estimate_project(self, selected_variables: List[str], variable_values: Dict[str, Any]) -> Dict[str, Any]:
        past_projects = self.get_past_projects()
    
        filtered_values = {var: variable_values.get(var, "Not specified") for var in selected_variables}
    
        messages = [
            {"role": "system", "content": "You are an AI assistant specialized in ETF project estimation."},
            {"role": "user", "content": f"""
            Given the following project variables, their values, and information about past projects:

            Current Project:
            {json.dumps(filtered_values, indent=2)}

            Past Projects:
            {json.dumps(past_projects, indent=2)}

            Provide an estimation of the project's complexity, duration, and cost. Explain your reasoning.
            Also, suggest any additional variables that might improve the estimation if provided.

            Consider the following factors specific to ETF projects:
            1. Type of ETF (e.g., equity, fixed income, commodity)
            2. Complexity of underlying assets (e.g., derivatives, options)
            3. Regulatory requirements and compliance considerations
            4. Technology infrastructure needs and integration challenges
            5. Market volatility and risk factors
            6. Data management and integration complexities
            7. Trading frequency and rebalancing requirements
            8. Historical performance analysis needs
            9. Competitor landscape and market positioning
            10. Liquidity considerations
            11. Tax efficiency and implications
            12. Marketing and distribution strategies
            13. Operational efficiency and scalability
            14. Investor education and transparency requirements

            Return your response in the following JSON format:
            {{
               "complexity": {{
                  "score": float,
                  "explanation": "string"
                }},
                "duration": {{
                   "estimate": "string",
                   "explanation": "string"
                }},
                "cost": {{
                   "estimate": "string",
                   "explanation": "string"
                }},
                "additional_variables": ["string"],
                "risk_factors": ["string"],
                "recommendations": ["string"],
                "historical_influence": "string",
                "team_composition": {{
                    "Project Manager": float,
                    "ETF Specialist": float,
                    "Data Engineer": float,
                    "Financial Analyst": float,
                    "Compliance Officer": float,
                    "Software Developer": float
                }},
                "etf_performance_factors": {{
                    "market_correlation": float,
                    "volatility": float,
                    "liquidity": float
                }}
            }}

            Ensure that the output is a valid JSON string without any markdown formatting or backticks.
            If you cannot provide an estimate due to insufficient information, please include an "error" field in the JSON with an explanation.
            """
            }
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages
            )
            result = response.choices[0].message.content
            logger.info(f"LLM response for project estimation: {result}")

            estimation_dict = json.loads(result)

            if "error" not in estimation_dict:
                if 'cost' in estimation_dict and 'estimate' in estimation_dict['cost']:
                    estimation_dict['cost']['parsed_estimate'] = self.parse_cost(estimation_dict['cost']['estimate'])
            
                estimation_dict = self.validate_and_adjust_estimation(estimation_dict, filtered_values)
                self.save_project(filtered_values, estimation_dict)
            return estimation_dict
    
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON in estimate_project: {e}")
            logger.error(f"Raw content: {result}")
            return {"error": f"Failed to parse estimation result. Raw content: {result}"}
        except KeyError as e:
            logger.error(f"KeyError in estimate_project: {e}")
            logger.error(f"Estimation dict: {estimation_dict}")
            return {"error": f"Missing key in estimation result: {e}"}
        except Exception as e:
            logger.error(f"An error occurred in estimate_project: {e}")
            return {"error": str(e)}
        
    def validate_and_adjust_estimation(self, estimation: Dict[str, Any], input_values: Dict[str, Any]) -> Dict[str, Any]:
        if 'complexity' in estimation and 'score' in estimation['complexity']:
            estimation['complexity']['score'] = max(0, min(10, estimation['complexity']['score']))

    # Handle duration
        if 'duration' not in estimation:
            estimation['duration'] = {}

        if 'start_date' in input_values and 'end_date' in input_values:
            start_date = self.parse_date(input_values['start_date'])
            end_date = self.parse_date(input_values['end_date'])
            if start_date and end_date:
                actual_duration = (end_date - start_date).days
                estimation['duration']['estimate'] = f"{actual_duration} days"
                estimation['duration']['explanation'] = f"Duration calculated based on provided start and end dates: {actual_duration} days."
            else:
                estimation['duration']['estimate'] = "Unable to calculate"
                estimation['duration']['explanation'] = "Could not calculate duration due to invalid date format."
        elif 'duration' in estimation and 'estimate' in estimation['duration']:
            # If duration estimate exists, ensure it's in the correct format
            try:
                duration_value = int(estimation['duration']['estimate'].split()[0])
                estimation['duration']['estimate'] = f"{duration_value} days"
            except ValueError:
                estimation['duration']['estimate'] = "Unable to parse"
                estimation['duration']['explanation'] = "Invalid duration format in estimation."
        else:
            estimation['duration']['estimate'] = "Not provided"
            estimation['duration']['explanation'] = "Duration not provided in input or estimation."

        # Handle cost
        if 'budget' in input_values and 'cost' in estimation:
            try:
                budget = self.parse_cost(input_values['budget'])
                cost_estimate = self.parse_cost(estimation['cost']['estimate'])
                
                if cost_estimate > budget:
                    estimation['cost']['estimate'] = f"${budget:,.2f}"
                    estimation['cost']['explanation'] += f" The estimate has been adjusted to not exceed the provided budget of ${budget:,.2f}."
                
                estimation['cost']['parsed_estimate'] = cost_estimate
            except ValueError as e:
                print(f"Warning: Could not adjust cost estimate due to invalid number format: {e}")
                estimation['cost']['parsed_estimate'] = estimation['cost'].get('estimate', 0)

        return estimation

    def generate_risk_matrix(self) -> Dict[str, List[float]]:
        return {
            'Impact': [1, 2, 3, 2, 4, 6, 3, 6, 9],
            'Likelihood': [1, 1, 1, 2, 2, 2, 3, 3, 3],
            'Risk Level': [1, 2, 3, 2, 4, 6, 3, 6, 9]
        }

    

    def simulate_etf_performance(self, days: int = 365) -> Dict[str, List[float]]:
        performance = 100 + np.cumsum(np.random.normal(0, 1, days))
        return {
            'days': list(range(days)),
            'value': performance.tolist()
        }

    def generate_sensitivity_analysis(self) -> Dict[str, List[float]]:
        factors = ['Market Volatility', 'Interest Rates', 'Regulatory Changes']
        impact = [0.3, 0.2, 0.5]
        return {'factors': factors, 'impact': impact}

    def answer_followup_question(self, question: str, project_context: Dict[str, Any]) -> str:
        messages = [
            {"role": "system", "content": "You are an AI assistant specialized in ETF project management."},
            {"role": "user", "content": f"""
            Given the following project context and a follow-up question, provide a detailed and relevant answer:

            Project Context:
            {json.dumps(project_context, indent=2)}

            Follow-up Question:
            {question}

            Please provide a comprehensive answer to the question, considering the project context and any relevant ETF-specific knowledge.
            """}
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"An error occurred while answering the follow-up question: {e}")
            return "I apologize, but I encountered an error while processing your question. Please try asking again or rephrase your question."

    def simulate_project_scenarios(self, base_duration: int, base_cost: float, market_condition: float, regulatory_change: bool) -> Dict[str, float]:
        impact = market_condition * 0.5 + (0.2 if regulatory_change else 0)
        adjusted_duration = base_duration * (1 + impact)
        adjusted_cost = base_cost * (1 + impact)
        return {
            'adjusted_duration': adjusted_duration,
            'adjusted_cost': adjusted_cost
        }

class ETFOptionsData:
    def __init__(self):
        self.data = {}

    def fetch_options_data(self, symbol):
        ticker = yf.Ticker(symbol)
        options = ticker.options
        
        if options:
            expiration = options[0]
            opt = ticker.option_chain(expiration)
            
            self.data[symbol] = {
                'calls': opt.calls,
                'puts': opt.puts,
                'underlying_price': ticker.info['regularMarketPrice']
            }
            return True
        return False

    def get_options_complexity(self, symbol):
        if symbol not in self.data:
            if not self.fetch_options_data(symbol):
                return 0
        
        options_data = self.data[symbol]
        num_options = len(options_data['calls']) + len(options_data['puts'])
        
        if num_options > 100:
            return 2
        elif num_options > 50:
            return 1
        else:
            return 0

class ETFProjectManagementAI:
    def __init__(self, input_size=25):
        self.scaler = StandardScaler()
        self.scaler_fitted = False
        self.document_processor = LLMAssistedProjectEstimator()
        self.risk_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.effort_estimator = GradientBoostingRegressor(n_estimators=100, random_state=42)
       # self.logger = logging.getLogger(__name__)
        
        self.input_size = input_size
        self.team_generator = nn.Sequential(
            nn.Linear(self.input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 7),
            nn.Softmax(dim=1)
        )
        
        #logger.info(f"Initialized ETFProjectManagementAI with input size: {self.input_size}")
       # logger.info(f"Team Generator Architecture: {self.team_generator}")

    def extract_project_details(self, document):
        extracted_info = self.document_processor.extract_initial_info(document)
        
        project_params = {
            'duration': max((extracted_info['end_date'] - extracted_info['start_date']).days, 1) if extracted_info['start_date'] and extracted_info['end_date'] else 180,
            'budget': max(extracted_info['budget'], 1), 
            'team_size': len(extracted_info['resource_costs']) or 1, 
            'industry': 'Finance',
            'project_type': 'Software',
            'etf_type': 'Active',
            'asset_class': 'Equity',
            'num_underlying_assets': 100,
            'rebalancing_frequency': 'Daily',
            'regulatory_regime': 'SEC',
            'data_sources': 1,
            'total_resource_cost': sum(extracted_info['resource_costs'].values()) or 1,
            'max_resource_cost': max(extracted_info['resource_costs'].values()) if extracted_info['resource_costs'] else 1,
            'min_resource_cost': min(extracted_info['resource_costs'].values()) if extracted_info['resource_costs'] else 1,
            'stakeholders': max(extracted_info['stakeholders'], 1),  
        }
        
        return project_params

    def preprocess_data(self, data, fit=False):
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame([data])

        expected_columns = [
            'duration', 'budget', 'team_size', 'manager_experience', 'num_underlying_assets', 
            'data_sources', 'stakeholders', 'complexity_Low', 'complexity_Medium', 'complexity_High',
            'industry_Tech', 'industry_Finance', 'industry_Healthcare', 'industry_Retail',
            'project_type_Software', 'project_type_Hardware', 'project_type_Service', 'project_type_Research',
            'etf_type_Index', 'etf_type_Active', 'etf_type_Smart Beta',
            'asset_class_Equity', 'asset_class_Fixed Income', 'asset_class_Commodity', 'asset_class_Multi-Asset',
            'rebalancing_frequency_Daily', 'rebalancing_frequency_Weekly', 'rebalancing_frequency_Monthly', 'rebalancing_frequency_Quarterly',
            'regulatory_regime_SEC', 'regulatory_regime_UCITS', 'regulatory_regime_Both'
        ]

        for col in expected_columns:
            if col not in data.columns:
                data[col] = 0

        data = data.reindex(columns=expected_columns)

        if fit or not self.scaler_fitted:
            self.scaler.fit(data)
            self.scaler_fitted = True

        scaled_data = self.scaler.transform(data)
        preprocessed_data = pd.DataFrame(scaled_data, columns=data.columns)

        return preprocessed_data

    def estimate_etf_complexity(self, project_features):
        if isinstance(project_features, pd.DataFrame):
            project_features = project_features.iloc[0]
    
        complexity_score = 0
    
        num_assets = project_features.get('num_underlying_assets', 0)
        if isinstance(num_assets, pd.Series):
            num_assets = num_assets.iloc[0] if not num_assets.empty else 0
        complexity_score += num_assets / 1000
    
        daily_rebalancing = project_features.get('rebalancing_frequency_Daily', 0)
        if isinstance(daily_rebalancing, pd.Series):
            daily_rebalancing = daily_rebalancing.iloc[0] if not daily_rebalancing.empty else 0
        complexity_score += (daily_rebalancing == 1) * 0.5
    
        data_sources = project_features.get('data_sources', 0)
        if isinstance(data_sources, pd.Series):
            data_sources = data_sources.iloc[0] if not data_sources.empty else 0
        complexity_score += data_sources / 10
    
        etf_type_active = project_features.get('etf_type_Active', 0)
        if isinstance(etf_type_active, pd.Series):
            etf_type_active = etf_type_active.iloc[0] if not etf_type_active.empty else 0
        if etf_type_active == 1:
            complexity_score += 0.3
    
        etf_type_smart_beta = project_features.get('etf_type_Smart Beta', 0)
        if isinstance(etf_type_smart_beta, pd.Series):
            etf_type_smart_beta = etf_type_smart_beta.iloc[0] if not etf_type_smart_beta.empty else 0
        if etf_type_smart_beta == 1:
            complexity_score += 0.2
    
        asset_class_multi = project_features.get('asset_class_Multi-Asset', 0)
        if isinstance(asset_class_multi, pd.Series):
            asset_class_multi = asset_class_multi.iloc[0] if not asset_class_multi.empty else 0
        if asset_class_multi == 1:
            complexity_score += 0.2
    
        asset_class_fixed = project_features.get('asset_class_Fixed Income', 0)
        if isinstance(asset_class_fixed, pd.Series):
            asset_class_fixed = asset_class_fixed.iloc[0] if not asset_class_fixed.empty else 0
        if asset_class_fixed == 1:
            complexity_score += 0.1
    
        complexity_score = min(max(complexity_score, 0), 1)
    
        complexity_dict = {
            'Low': max(0, 1 - complexity_score),
            'Medium': max(0, min(complexity_score, 1 - complexity_score)),
            'High': max(0, complexity_score - 0.5)
        }
    
        return complexity_dict, complexity_score

    def classify_risk(self, project_features):
        if isinstance(project_features, pd.DataFrame):
            project_features = project_features.iloc[0]

        duration = project_features.get('duration', 0)
        budget = project_features.get('budget', 0)
        team_size = project_features.get('team_size', 0)
        total_resource_cost = project_features.get('total_resource_cost', budget)
        max_resource_cost = project_features.get('max_resource_cost', budget)
        min_resource_cost = project_features.get('min_resource_cost', budget / team_size if team_size else 0)

        duration_factor = duration / 365 if duration else 0
        budget_factor = budget / 10000000 if budget else 0
        team_size_factor = team_size / 20 if team_size else 0
        resource_cost_factor = total_resource_cost / budget if budget else 0
        resource_cost_disparity = (max_resource_cost / min_resource_cost - 1) if min_resource_cost else 0

        risk_score = (duration_factor + budget_factor + team_size_factor + resource_cost_factor + resource_cost_disparity)
        risk_score = min(max(risk_score / 5, 0), 1)

        risk_dict = {
            'Low': max(0, 1 - risk_score),
            'Medium': max(0, min(risk_score, 1 - risk_score)),
            'High': max(0, risk_score - 0.5)
        }

        return risk_dict, risk_score

    def estimate_effort(self, project_features):
        if isinstance(project_features, pd.DataFrame):
            project_features = project_features.iloc[0]

        duration = project_features.get('duration', 180)
        team_size = project_features.get('team_size', 5)
        num_underlying_assets = project_features.get('num_underlying_assets', 100)
        stakeholders = project_features.get('stakeholders', 3)

        effort = (duration * 
                 (1 + team_size / 10) * 
                 (1 + num_underlying_assets / 1000) *
                 (1 + stakeholders / 10))

        return effort

    def generate_team(self, project_features):
        if isinstance(project_features, pd.DataFrame):
            input_tensor = torch.FloatTensor(project_features.values)
        elif isinstance(project_features, pd.Series):
            input_tensor = torch.FloatTensor(project_features.values.reshape(1, -1))
        else:
            input_tensor = torch.FloatTensor(project_features)

        if input_tensor.dim() == 1:
           input_tensor = input_tensor.unsqueeze(0)

        expected_input_size = self.team_generator[0].in_features
        actual_input_size = input_tensor.shape[1]

        if actual_input_size != expected_input_size:
            if actual_input_size > expected_input_size:
                input_tensor = input_tensor[:, :expected_input_size]
            else:
                padding = torch.zeros(input_tensor.shape[0], expected_input_size - actual_input_size)
                input_tensor = torch.cat([input_tensor, padding], dim=1)

        with torch.no_grad():
            team_composition = self.team_generator(input_tensor)
            team_composition = F.softmax(team_composition, dim=1)

        roles = ['Project Manager', 'ETF Specialist', 'Data Engineer', 'Financial Analyst', 'Compliance Officer', 'Software Developer', 'QA Tester']
    
        team_composition = team_composition.numpy()[0]
    
        min_allocation = 0.05
        team_composition = np.maximum(team_composition, min_allocation)
        team_composition /= team_composition.sum()
    
        if 'etf_type_Active' in project_features.columns and project_features['etf_type_Active'].iloc[0] == 1:
            team_composition[roles.index('ETF Specialist')] *= 1.3
            team_composition[roles.index('Financial Analyst')] *= 1.2
        if 'regulatory_regime_Both' in project_features.columns and project_features['regulatory_regime_Both'].iloc[0] == 1:
            team_composition[roles.index('Compliance Officer')] *= 1.3
        if 'data_sources' in project_features.columns and project_features['data_sources'].iloc[0] > 5:
            team_composition[roles.index('Data Engineer')] *= 1.2
    
        team_composition /= team_composition.sum()
    
        return dict(zip(roles, team_composition))

    def predict_success(self, project_features, risk_score, complexity_score):
        if isinstance(project_features, pd.DataFrame):
            project_features = project_features.iloc[0]

        base_probability = 0.7
    
        risk_factor = 1 - risk_score
        complexity_factor = 1 - complexity_score
    
        budget = project_features.get('budget', 0)
        total_resource_cost = project_features.get('total_resource_cost', budget)
        budget_factor = min(budget / total_resource_cost, 1.5) if total_resource_cost > 0 else 1.0
    
        team_size = project_features.get('team_size', 5)
        stakeholders = project_features.get('stakeholders', 3)
        team_factor = min(team_size / 5, 1.5)
        stakeholder_factor = max(1 - (stakeholders - 3) / 10, 0.5)
    
        success_probability = base_probability * risk_factor * complexity_factor * budget_factor * team_factor * stakeholder_factor
        success_probability = max(min(success_probability, 0.99), 0.01)
    
        explanation = f"""
        Base probability: 70%
        Risk adjustment: {risk_factor:.2f}
        Complexity adjustment: {complexity_factor:.2f}
        Budget utilization adjustment: {budget_factor:.2f}
        Team size adjustment: {team_factor:.2f}
        Stakeholder adjustment: {stakeholder_factor:.2f}
        Final success probability: {success_probability:.2%}
        """
    
        return success_probability, explanation

    def simulate_etf_project(self, project_features, risk_score, success_probability, complexity_score, n_steps=12):
        if isinstance(project_features, pd.DataFrame):
            project_features = project_features.iloc[0]

        duration = project_features.get('duration', 180)
        budget = project_features.get('budget', 500000)
        total_resource_cost = project_features.get('total_resource_cost', budget)

        progress = [0]
        spent_budget = [0]
        current_progress = 0
        current_spent = 0

        base_progress_rate = 1 / duration

        for step in range(1, n_steps):
            expected_progress = min(1, step * base_progress_rate * n_steps)
            progress_variability = np.random.normal(1, 0.1 * risk_score)

            progress_increment = max(0.01, (expected_progress - current_progress) * progress_variability * success_probability)
            current_progress = min(1, current_progress + progress_increment)
            progress.append(current_progress)

            expected_spent = total_resource_cost * current_progress
            budget_variability = np.random.normal(1, 0.05 * risk_score)

            budget_increment = max(0.01 * budget, (expected_spent - current_spent) * budget_variability)
            current_spent = min(budget, current_spent + budget_increment)

            overspend_chance = risk_score * 0.2
            if np.random.random() < overspend_chance:
                current_spent *= 1.1

            spent_budget.append(current_spent)

        final_progress_adjustment = 1 - (risk_score * 0.2 + complexity_score * 0.2)
        progress = [p * final_progress_adjustment for p in progress]

        return {'progress': progress, 'spent_budget': spent_budget}
    
    def process_project_document(self, document):
        try:
           # self.logger.info("ETFProjectManagementAI processing document...")
            
            extracted_info = self.document_processor.extract_initial_info(document)
           # self.logger.info("Extracted info:")
            #self.logger.info(extracted_info)
        
            project_features = self.preprocess_data(self.extract_project_details(extracted_info))
           # self.logger.info("Converted project features:")
           # self.logger.info(project_features)
        
            complexity_dict, complexity_score = self.estimate_etf_complexity(project_features)
           # self.logger.info(f"Complexity: {complexity_dict}, Score: {complexity_score}")
        
            risk_dict, risk_score = self.classify_risk(project_features)
           # self.logger.info(f"Risk: {risk_dict}, Score: {risk_score}")
        
            effort_estimate = self.estimate_effort(project_features)
           # self.logger.info(f"Effort Estimate: {effort_estimate}")
        
            team_composition = self.generate_team(project_features)
          #  self.logger.info(f"Team Composition: {team_composition}")
        
            success_probability, explanation = self.predict_success(project_features, risk_score, complexity_score)
          #  self.logger.info(f"Success Probability: {success_probability}")
           # self.logger.info(f"Explanation: {explanation}")
        
                    
            return {
                'extracted_details': extracted_info,
                'complexity': (complexity_dict, complexity_score),
                'risk': (risk_dict, risk_score),
                'effort': effort_estimate,
                'team': team_composition,
                'success': (success_probability, explanation)
                
            }
        except Exception as e:
            self.logger.error(f"Error in processing document: {str(e)}")
            self.logger.error("Traceback:", exc_info=True)
            raise

    def generate_risk_matrix(self):
        return {
            'Impact': [1, 2, 3, 2, 4, 6, 3, 6, 9],
            'Likelihood': [1, 1, 1, 2, 2, 2, 3, 3, 3],
            'Risk Level': [1, 2, 3, 2, 4, 6, 3, 6, 9]
        }


    
    def generate_sensitivity_analysis(self):
        factors = ['Market Volatility', 'Interest Rates', 'Regulatory Changes']
        impact = [0.3, 0.2, 0.5]
        return {'factors': factors, 'impact': impact}

    def answer_followup_question(self, question, project_context):
        return self.document_processor.answer_followup_question(question, project_context)

    def simulate_project_scenarios(self, base_duration, base_cost, market_condition, regulatory_change):
        impact = market_condition * 0.5 + (0.2 if regulatory_change else 0)
        adjusted_duration = base_duration * (1 + impact)
        adjusted_cost = base_cost * (1 + impact)
        return {
            'adjusted_duration': adjusted_duration,
            'adjusted_cost': adjusted_cost
        }

# Example usage
if __name__ == "__main__":
    etf_pm_ai = ETFProjectManagementAI()
    document = "Your project document text here..."
    result = etf_pm_ai.process_project_document(document)
    print("Project Analysis Result:", result)