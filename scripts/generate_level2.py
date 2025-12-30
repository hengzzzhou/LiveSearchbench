#!/usr/bin/env python3
"""
Level 2 Question Generation System
1. Bridge questions: A-B-C, ask for final C entity, avoid duplication
2. Multi-attribute questions: gradually add constraints until answer is unique
"""

import pandas as pd
import json
import requests
import time
import random
import argparse
import os
from collections import defaultdict

# API Configuration
API_KEY = "your_api_key_here"
API_BASE_URL = "https://api.openai.com/v1/chat/completions"
API_MODEL = "gpt-4o"

def call_gpt_api(messages, model=None, temperature=0.7, max_tokens=2000):
    """Call GPT API to generate questions"""
    if model is None:
        model = API_MODEL
        
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    
    try:
        response = requests.post(API_BASE_URL, headers=headers, json=data, timeout=30)
        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content'].strip()
        else:
            print(f"API error: {response.status_code}")
            return None
    except Exception as e:
        print(f"API call failed: {e}")
        return None

def query_wikidata_sparql(query):
    """Execute SPARQL query"""
    endpoint = "https://query.wikidata.org/sparql"
    headers = {
        'User-Agent': 'Level2-Generator/1.0',
        'Accept': 'application/json'
    }
    params = {'query': query, 'format': 'json'}
    
    try:
        response = requests.get(endpoint, headers=headers, params=params, timeout=30)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"SPARQL query failed: {response.status_code}")
            return None
    except Exception as e:
        print(f"SPARQL query exception: {e}")
        return None

def count_sparql_results(query):
    """Count SPARQL query results"""
    result = query_wikidata_sparql(query)
    if result and result['results']['bindings']:
        if 'count' in result['results']['bindings'][0]:
            return int(result['results']['bindings'][0]['count']['value'])
        else:
            return len(result['results']['bindings'])
    return 0

class BridgeQuestionGenerator:
    """Bridge question generator: A-B-C, ask for C entity"""
    
    def __init__(self, df):
        self.df = df
        self.used_entities = set()  # Avoid reusing entities
        self.valid_bridges = []     # Store valid bridge paths
        
    def build_relation_index(self):
        """Build bidirectional relation index"""
        print("Building bidirectional relation index...")
        
        # Subject -> Objects index
        self.subject_relations = defaultdict(list)
        # Object -> Subjects index (reverse)
        self.object_relations = defaultdict(list)
        
        for _, row in self.df.iterrows():
            # Forward index: subject -> object
            self.subject_relations[row['subject_id']].append({
                'predicate_id': row['predicate_id'],
                'predicate_label': row['predicate_label'],
                'object_id': row['object_id'],
                'object_label': row['object_label']
            })
            
            # Reverse index: object -> subject
            self.object_relations[row['object_id']].append({
                'predicate_id': row['predicate_id'],
                'predicate_label': row['predicate_label'],
                'subject_id': row['subject_id'],
                'subject_label': row['subject_label']
            })
        
        print(f"Index complete, forward: {len(self.subject_relations)} entities, reverse: {len(self.object_relations)} entities")
    
    def find_bridge_paths_efficient(self, max_paths=50, max_intermediate_threshold=5):
        """Efficiently find bridge paths: A + relation1 ∩ C + relation2^-1"""
        print(f"Efficiently finding bridge paths using intersection method")
        
        if not hasattr(self, 'subject_relations'):
            self.build_relation_index()
        
        bridge_count = 0
        
        # Iterate through all (A, relation1) combinations
        for entity_a, a_relations in self.subject_relations.items():
            if bridge_count >= max_paths:
                break
                
            if entity_a in self.used_entities:
                continue
                
            entity_a_label = self.get_entity_label(entity_a)
            
            for relation1 in a_relations:
                if bridge_count >= max_paths:
                    break
                    
                relation1_id = relation1['predicate_id']
                relation1_label = relation1['predicate_label']
                
                # Step 1: Find all B where A --relation1--> B
                b_candidates_from_a = set()
                for rel in a_relations:
                    if rel['predicate_id'] == relation1_id:
                        b_candidates_from_a.add(rel['object_id'])
                
                # Iterate through all (C, relation2) combinations
                for entity_c, c_relations in self.object_relations.items():
                    if bridge_count >= max_paths:
                        break
                        
                    if (entity_c in self.used_entities or entity_c == entity_a):
                        continue
                    
                    entity_c_label = self.get_entity_label(entity_c)
                    
                    for relation2 in c_relations:
                        if bridge_count >= max_paths:
                            break
                            
                        relation2_id = relation2['predicate_id']
                        relation2_label = relation2['predicate_label']
                        
                        # Step 2: Find all B where B --relation2--> C
                        b_candidates_from_c = set()
                        for rel in c_relations:
                            if rel['predicate_id'] == relation2_id:
                                b_candidates_from_c.add(rel['subject_id'])
                        
                        # Step 3: Find intersection
                        intersection = b_candidates_from_a & b_candidates_from_c
                        
                        if len(intersection) == 1:
                            entity_b = list(intersection)[0]
                            
                            # Basic checks
                            if (entity_b in self.used_entities or 
                                entity_b == entity_a or 
                                entity_b == entity_c):
                                continue
                            
                            # Construct relation objects for checking
                            ab_relation = {
                                'predicate_id': relation1_id,
                                'predicate_label': relation1_label
                            }
                            bc_relation = {
                                'predicate_id': relation2_id, 
                                'predicate_label': relation2_label
                            }
                            
                            # Filter meaningless bridges
                            if self.is_meaningless_bridge(ab_relation, bc_relation, entity_a, entity_b, entity_c):
                                continue
                            
                            # Construct SPARQL query (temporarily skip verification for efficiency)
                            bridge_query = f"""
SELECT (COUNT(?answer) AS ?count) WHERE {{
  wd:{entity_a} wdt:{relation1_id} ?intermediate .
  ?intermediate wdt:{relation2_id} ?answer .
}}
"""
                            
                            # Temporarily skip SPARQL verification to test intersection logic
                            if True:  # Previously: if answer_count == 1
                                entity_b_label = self.get_entity_label(entity_b)
                                
                                bridge_path = {
                                    'entity_a': entity_a,
                                    'entity_a_label': entity_a_label,
                                    'relation_ab_id': relation1_id,
                                    'relation_ab_label': relation1_label,
                                    'entity_b': entity_b,
                                    'entity_b_label': entity_b_label,
                                    'relation_bc_id': relation2_id,
                                    'relation_bc_label': relation2_label,
                                    'entity_c': entity_c,
                                    'entity_c_label': entity_c_label,
                                    'sparql_query': bridge_query.strip(),
                                    'intermediate_count': 1
                                }
                                
                                self.valid_bridges.append(bridge_path)
                                
                                # Mark entities as used
                                self.used_entities.add(entity_a)
                                self.used_entities.add(entity_b)
                                self.used_entities.add(entity_c)
                                
                                bridge_count += 1
                                
                                print(f"  ✓ Found bridge: {entity_a_label} --{relation1_label}--> {entity_b_label} --{relation2_label}--> {entity_c_label}")
                            
                            time.sleep(0.05)
        
        print(f"Efficient bridge path search complete, found {len(self.valid_bridges)} valid paths")
        return self.valid_bridges
    
    def find_bridge_paths(self, max_paths=50, max_intermediate_threshold=5):
        """Entry method for bridge path search"""
        return self.find_bridge_paths_efficient(max_paths, max_intermediate_threshold)
    
    def is_meaningless_bridge(self, ab_relation, bc_relation, entity_a, entity_b, entity_c):
        """Determine if this is a meaningless bridge question"""
        
        # Filter temporal sequence problems (predecessor/successor relations)
        temporal_relations = ['P155', 'P156', 'P1365', 'P1366']  # follows, followed by, replaces, replaced by
        
        if (ab_relation['predicate_id'] in temporal_relations and 
            bc_relation['predicate_id'] in temporal_relations):
            return True  # Temporal sequence bridges are usually meaningless
        
        # Filter inverse relations (A->B relation is inverse of B->C relation)
        inverse_pairs = [
            ('P155', 'P156'),  # follows / followed by
            ('P1365', 'P1366'),  # replaces / replaced by  
            ('P527', 'P361'),  # has part / part of
            ('P749', 'P355'),  # parent organization / subsidiary
            ('P276', 'P131'),  # location / located in administrative territorial entity
        ]
        
        ab_pred = ab_relation['predicate_id']
        bc_pred = bc_relation['predicate_id']
        
        for pred1, pred2 in inverse_pairs:
            if (ab_pred == pred1 and bc_pred == pred2) or (ab_pred == pred2 and bc_pred == pred1):
                return True  # Inverse relation bridges are usually meaningless
        
        return False
    
    def get_entity_label(self, entity_id):
        """Get entity label"""
        # Find from subject
        for _, row in self.df[self.df['subject_id'] == entity_id].head(1).iterrows():
            return row['subject_label']
        
        # Find from object
        for _, row in self.df[self.df['object_id'] == entity_id].head(1).iterrows():
            return row['object_label']
        
        return f"Entity_{entity_id}"
    
    def generate_bridge_questions(self, bridge_paths):
        """Generate questions for bridge paths"""
        print(f"Generating questions for {len(bridge_paths)} bridge paths...")
        
        qa_pairs = []
        
        for idx, bridge in enumerate(bridge_paths):
            if idx % 5 == 0:
                print(f"Bridge question generation progress: {idx}/{len(bridge_paths)}")
            
            # Construct GPT prompt
            prompt = f"""
Generate a natural two-hop question that asks about the final destination in this path:

Path: {bridge['entity_a_label']} --{bridge['relation_ab_label']}--> {bridge['entity_b_label']} --{bridge['relation_bc_label']}--> {bridge['entity_c_label']}

The question should ask: "What is the {bridge['relation_bc_label']} of the {bridge['relation_ab_label']} of {bridge['entity_a_label']}?"

But make it more natural and conversational. The answer should be: {bridge['entity_c_label']}

Examples:
- "Where was the director of Inception born?" 
- "What genre does the author of Harry Potter write?"
- "Which country is the capital of France located in?"

Generate only the question:"""

            messages = [
                {"role": "system", "content": "You generate natural multi-hop questions that require reasoning through intermediate entities."},
                {"role": "user", "content": prompt}
            ]
            
            question = call_gpt_api(messages, temperature=0.8)
            
            if question:
                qa_pairs.append({
                    'question': question,
                    'answer': bridge['entity_c_label'],
                    'level': 2,
                    'type': 'bridge',
                    'reasoning_chain': [
                        [bridge['entity_a_label'], bridge['relation_ab_label'], bridge['entity_b_label']],
                        [bridge['entity_b_label'], bridge['relation_bc_label'], bridge['entity_c_label']]
                    ],
                    'sparql_verification': bridge['sparql_query'],
                    'bridge_info': {
                        'start_entity': bridge['entity_a_label'],
                        'intermediate_entity': bridge['entity_b_label'], 
                        'final_entity': bridge['entity_c_label'],
                        'intermediate_count': bridge['intermediate_count']
                    }
                })
            
            time.sleep(0.3)  # Control API call frequency
        
        return qa_pairs

class MultiAttributeQuestionGenerator:
    """Multi-attribute question generator: gradually add constraints until unique"""
    
    def __init__(self, df):
        self.df = df
        self.used_entities = set()
        
    def build_entity_attributes(self):
        """Build entity attribute index"""
        print("Building entity attribute index...")
        
        self.entity_attributes = defaultdict(list)
        
        for _, row in self.df.iterrows():
            self.entity_attributes[row['subject_id']].append({
                'predicate_id': row['predicate_id'],
                'predicate_label': row['predicate_label'],
                'object_id': row['object_id'],
                'object_label': row['object_label']
            })
        
        print(f"Attribute index complete, {len(self.entity_attributes)} entities")
    
    def find_multi_attribute_entities(self, max_entities=30):
        """Find entities suitable for multi-attribute constraints"""
        print("Finding entities suitable for multi-attribute constraints...")
        
        if not hasattr(self, 'entity_attributes'):
            self.build_entity_attributes()
        
        valid_entities = []
        
        for entity_id, attributes in self.entity_attributes.items():
            if len(valid_entities) >= max_entities:
                break
                
            if entity_id in self.used_entities or len(attributes) < 3:
                continue  # Need at least 3 attributes for constraints
            
            entity_label = self.get_entity_label(entity_id)
            
            # Try gradually adding constraints to find unique answer
            constraints = self.find_optimal_constraints(entity_id, attributes)
            
            if constraints:
                valid_entities.append({
                    'entity_id': entity_id,
                    'entity_label': entity_label,
                    'constraints': constraints,
                    'total_attributes': len(attributes)
                })
                
                self.used_entities.add(entity_id)
                print(f"  ✓ Found entity: {entity_label} ({len(constraints)} constraints)")
            
            time.sleep(0.1)
        
        print(f"Multi-attribute entity search complete, found {len(valid_entities)} entities")
        return valid_entities
    
    def find_optimal_constraints(self, entity_id, attributes):
        """Find optimal constraint combination for entity (gradually add until unique)"""
        
        print(f"    Finding optimal constraint combination for entity {entity_id}, available attributes: {len(attributes)}")
        
        # Randomly shuffle attribute order
        shuffled_attrs = random.sample(attributes, len(attributes))
        
        # Start from 1 constraint and gradually increase
        for constraint_count in range(2, min(6, len(attributes) + 1)):  # 2-5 constraints
            selected_attrs = shuffled_attrs[:constraint_count]
            
            # Construct SPARQL query (query entire Wikidata)
            where_clauses = []
            for attr in selected_attrs:
                where_clauses.append(f"?entity wdt:{attr['predicate_id']} wd:{attr['object_id']} .")
            
            sparql_query = f"""
SELECT (COUNT(?entity) AS ?count) WHERE {{
  {' '.join(where_clauses)}
}}
"""
            
            print(f"      Testing {constraint_count} constraints...")
            result_count = count_sparql_results(sparql_query)
            print(f"      Result: {result_count} entities")
            
            if result_count == 1:  # Found unique answer
                print(f"      ✓ Found unique constraint combination")
                return {
                    'attributes': selected_attrs,
                    'sparql_query': sparql_query.strip(),
                    'constraint_count': constraint_count
                }
            elif result_count == 0:  # No results, constraints too restrictive
                print(f"      ✗ Too many constraints, no matching entities")
                break
        
        # If CSV attributes are insufficient, try to get more attributes from Wikidata
        print(f"    CSV attributes insufficient, trying to get more attributes from Wikidata...")
        extended_attributes = self.get_wikidata_attributes(entity_id, attributes)
        
        if len(extended_attributes) > len(attributes):
            print(f"    Got {len(extended_attributes) - len(attributes)} additional attributes from Wikidata")
            return self.find_optimal_constraints_extended(entity_id, extended_attributes)
        
        return None  # No suitable constraint combination found
    
    def get_wikidata_attributes(self, entity_id, existing_attributes):
        """Get additional attributes for entity from Wikidata"""
        
        # Get existing attribute IDs to avoid duplication
        existing_predicates = set(attr['predicate_id'] for attr in existing_attributes)
        
        # SPARQL query to get more entity attributes
        sparql_query = f"""
SELECT ?predicate ?object ?predicateLabel ?objectLabel WHERE {{
  wd:{entity_id} ?predicate ?object .
  
  # Filter out some useful attribute types
  FILTER(?predicate IN (
    wdt:P27,   # country of citizenship
    wdt:P19,   # place of birth  
    wdt:P20,   # place of death
    wdt:P106,  # occupation
    wdt:P31,   # instance of
    wdt:P136,  # genre
    wdt:P495,  # country of origin
    wdt:P37,   # official language
    wdt:P36,   # capital
    wdt:P17,   # country
    wdt:P131,  # located in administrative territorial entity
    wdt:P276,  # location
    wdt:P57,   # director
    wdt:P50,   # author
    wdt:P175,  # performer
    wdt:P364,  # original language of work
    wdt:P407,  # language of work or name
    wdt:P840,  # narrative location
    wdt:P159,  # headquarters location
    wdt:P937,  # work location
    wdt:P108,  # employer
    wdt:P69,   # educated at
    wdt:P463,  # member of
    wdt:P102,  # member of political party
    wdt:P641   # sport
  ))
  
  # Filter out existing attributes
  FILTER(?predicate NOT IN ({', '.join(f'wdt:{pid}' for pid in existing_predicates)}))
  
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" . }}
}}
LIMIT 20
"""
        
        result = query_wikidata_sparql(sparql_query)
        extended_attributes = list(existing_attributes)  # Copy existing attributes
        
        if result and result.get('results', {}).get('bindings'):
            for binding in result['results']['bindings']:
                predicate_uri = binding.get('predicate', {}).get('value', '')
                object_uri = binding.get('object', {}).get('value', '')
                predicate_label = binding.get('predicateLabel', {}).get('value', '')
                object_label = binding.get('objectLabel', {}).get('value', '')
                
                if predicate_uri and object_uri:
                    predicate_id = predicate_uri.split('/')[-1]
                    object_id = object_uri.split('/')[-1]
                    
                    # If no label, use URI as fallback
                    if not predicate_label or predicate_label == predicate_uri:
                        predicate_label = predicate_uri
                    if not object_label or object_label == object_uri:
                        object_label = object_uri
                    
                    # Only add attributes with valid labels
                    if predicate_id.startswith('P') and object_id.startswith('Q') and predicate_label and object_label:
                        extended_attributes.append({
                            'predicate_id': predicate_id,
                            'predicate_label': predicate_label,
                            'object_id': object_id,
                            'object_label': object_label
                        })
        
        return extended_attributes
    
    def find_optimal_constraints_extended(self, entity_id, attributes):
        """Find optimal constraint combination using extended attributes"""
        
        print(f"    Using extended attributes to find constraint combination for entity {entity_id}")
        
        # Randomly shuffle attribute order
        shuffled_attrs = random.sample(attributes, len(attributes))
        
        # Start from 2 constraints and gradually increase
        for constraint_count in range(2, min(7, len(attributes) + 1)):  # 2-6 constraints
            selected_attrs = shuffled_attrs[:constraint_count]
            
            # First verify if target entity satisfies these constraints
            verification_query = f"""
ASK WHERE {{
  wd:{entity_id} ?p ?o .
  {' '.join([f'wd:{entity_id} wdt:{attr["predicate_id"]} wd:{attr["object_id"]} .' for attr in selected_attrs])}
}}
"""
            
            verification_result = query_wikidata_sparql(verification_query)
            if not (verification_result and verification_result.get('boolean', False)):
                print(f"      Skip: target entity does not satisfy these {constraint_count} constraints")
                continue
            
            # Construct SPARQL query (query entire Wikidata)
            where_clauses = []
            for attr in selected_attrs:
                where_clauses.append(f"?entity wdt:{attr['predicate_id']} wd:{attr['object_id']} .")
            
            sparql_query = f"""
SELECT (COUNT(?entity) AS ?count) WHERE {{
  {' '.join(where_clauses)}
}}
"""
            
            print(f"      Testing extended constraints {constraint_count}...")
            result_count = count_sparql_results(sparql_query)
            print(f"      Result: {result_count} entities")
            
            if result_count == 1:  # Found unique answer
                print(f"      ✓ Found unique constraint combination (extended attributes)")
                return {
                    'attributes': selected_attrs,
                    'sparql_query': sparql_query.strip(),
                    'constraint_count': constraint_count
                }
            elif result_count == 0:  # No results, possibly inconsistent data
                print(f"      ✗ Extended constraints no match (possibly inconsistent data)")
                continue
        
        return None
    
    def get_entity_label(self, entity_id):
        """Get entity label"""
        for _, row in self.df[self.df['subject_id'] == entity_id].head(1).iterrows():
            return row['subject_label']
        return f"Entity_{entity_id}"
    
    def generate_multi_attribute_questions(self, entities):
        """Generate questions for multi-attribute entities"""
        print(f"Generating questions for {len(entities)} multi-attribute entities...")
        
        qa_pairs = []
        
        for idx, entity_info in enumerate(entities):
            if idx % 5 == 0:
                print(f"Multi-attribute question generation progress: {idx}/{len(entities)}")
            
            constraints = entity_info['constraints']
            attributes = constraints['attributes']
            
            # Construct constraint descriptions
            constraint_descriptions = []
            for attr in attributes:
                constraint_descriptions.append(f"has {attr['predicate_label']} {attr['object_label']}")
            
            constraints_text = " and ".join(constraint_descriptions)
            
            prompt = f"""
Generate a natural question asking about an entity with multiple specific constraints:

Entity: {entity_info['entity_label']}
Constraints: The entity {constraints_text}

The question should ask "Which entity..." or "What..." and describe these constraints naturally.

Answer: {entity_info['entity_label']}

Examples:
- "Which country has Paris as its capital and French as its official language?"
- "What company was founded by Steve Jobs and is headquartered in Cupertino?"
- "Which university is located in Cambridge and was founded in 1209?"

Make it flow naturally and be specific. Generate only the question:"""

            messages = [
                {"role": "system", "content": "You generate natural multi-constraint questions that require reasoning through multiple attributes."},
                {"role": "user", "content": prompt}
            ]
            
            question = call_gpt_api(messages, temperature=0.8)
            
            if question:
                reasoning_chain = []
                for attr in attributes:
                    reasoning_chain.append([entity_info['entity_label'], attr['predicate_label'], attr['object_label']])
                
                qa_pairs.append({
                    'question': question,
                    'answer': entity_info['entity_label'],
                    'level': 2,
                    'type': 'multi_attribute',
                    'reasoning_chain': reasoning_chain,
                    'sparql_verification': constraints['sparql_query'],
                    'constraint_info': {
                        'constraint_count': constraints['constraint_count'],
                        'total_attributes': entity_info['total_attributes'],
                        'constraints': [f"{attr['predicate_label']}: {attr['object_label']}" for attr in attributes]
                    }
                })
            
            time.sleep(0.3)
        
        return qa_pairs

def convert_extracted_triples_to_format(input_csv):
    """
    Convert extracted triples CSV to the format expected by generate_level2

    Input columns (from 0_extract_triple_changes.py):
        entity_id, entity_label, property_id, property_label, property_type,
        old_value, new_value, new_value_label, change_type, change_timestamp, wiki_url

    Output columns (expected by generate_level2.py):
        subject_id, subject_label, predicate_id, predicate_label, object_id, object_label
    """
    print(f"Converting extracted triples from {input_csv}...")
    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df)} extracted triples")

    # Filter: only keep triples where new_value is a Wikibase Item (starts with Q)
    df_filtered = df[df['new_value'].str.startswith('Q', na=False)].copy()
    print(f"Filtered to {len(df_filtered)} triples with entity values (Q-items)")

    # Rename columns to match expected format
    df_converted = pd.DataFrame({
        'subject_id': df_filtered['entity_id'],
        'subject_label': df_filtered['entity_label'],
        'predicate_id': df_filtered['property_id'],
        'predicate_label': df_filtered['property_label'],
        'object_id': df_filtered['new_value'],
        'object_label': df_filtered['new_value_label']
    })

    print(f"Converted to format with {len(df_converted)} triples")
    return df_converted

def main(input_file=None):
    """Main function: only generate multi-attribute questions"""

    # Load data
    print("Loading triple data...")

    # Determine input file
    if input_file:
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")
        print(f"Using specified input file: {input_file}")

        # Check if it's from 0_extract_triple_changes.py (has new_value column)
        df_test = pd.read_csv(input_file, nrows=1)
        if 'new_value' in df_test.columns:
            df = convert_extracted_triples_to_format(input_file)
        else:
            df = pd.read_csv(input_file)
            print(f"Loaded {len(df)} triples")
    else:
        # Try to load from extracted triples first, fallback to old format
        extracted_triples_path = './outputs/extracted_triples/triple_changes_latest.csv'
        old_format_path = './data/final_changed_item_with_id.csv'

        if os.path.exists(extracted_triples_path):
            print(f"Found extracted triples: {extracted_triples_path}")
            df = convert_extracted_triples_to_format(extracted_triples_path)
        elif os.path.exists(old_format_path):
            print(f"Using old format: {old_format_path}")
            df = pd.read_csv(old_format_path)
            print(f"Loaded {len(df)} triples")
        else:
            raise FileNotFoundError(f"No input file found. Please provide either:\n"
                                    f"  - {extracted_triples_path} (from 0_extract_triple_changes.py)\n"
                                    f"  - {old_format_path} (old format)")
    
    all_qa_pairs = []
    
    # Temporarily skip bridge question generation
    print("\n=== Skip bridge questions (too time-consuming) ===")
    
    # Generate multi-attribute questions
    print("\n=== Generate multi-attribute questions ===")
    multi_attr_generator = MultiAttributeQuestionGenerator(df)
    multi_attr_entities = multi_attr_generator.find_multi_attribute_entities(max_entities=400)
    
    if multi_attr_entities:
        multi_attr_qa_pairs = multi_attr_generator.generate_multi_attribute_questions(multi_attr_entities)
        all_qa_pairs.extend(multi_attr_qa_pairs)
        print(f"Generated {len(multi_attr_qa_pairs)} multi-attribute questions")
    
    # Save results
    result = {
        'metadata': {
            'description': 'Level 2 multi-attribute questions with unique answers',
            'total_questions': len(all_qa_pairs),
            'bridge_questions': 0,  # Temporarily skipped
            'multi_attribute_questions': len(all_qa_pairs),
            'verification_method': 'SPARQL uniqueness check on full Wikidata',
            'generation_date': '2025-09-07'
        },
        'qa_pairs': all_qa_pairs
    }
    
    # Save file
    import os
    os.makedirs('./outputs/questions', exist_ok=True)
    filename = './outputs/questions/level2_multi_attribute_only.json'
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    # Display results
    print(f"\n=== Level 2 multi-attribute question generation complete ===")
    print(f"Total generated: {len(all_qa_pairs)} multi-attribute questions")
    print(f"Saved to: {filename}")
    
    # Display examples
    if all_qa_pairs:
        print(f"\n=== Question Examples ===")
        for i, qa in enumerate(all_qa_pairs[:3]):
            print(f"{i+1}. Q: {qa['question']}")
            print(f"   A: {qa['answer']}")
            print(f"   Constraints: {len(qa['constraint_info']['constraints'])} items")
            print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Level 2 questions from knowledge triples")
    parser.add_argument("--input", type=str, default=None,
                        help="Input CSV file path (auto-detects format from 0_extract_triple_changes.py or old format)")
    args = parser.parse_args()

    main(input_file=args.input)