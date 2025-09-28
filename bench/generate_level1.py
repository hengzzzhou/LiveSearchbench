#!/usr/bin/env python3
"""
Level 1 Question Generation
Generate 300 Level 1 questions categorized by relation type, avoiding entity duplication
"""

import pandas as pd
import json
import requests
import time
import random
from collections import defaultdict

# API Configuration
API_KEY = "your_api_key_here"
API_BASE_URL = "https://api.openai.com/v1/chat/completions"
API_MODEL = "gpt-5"

def call_gpt_api(messages, model=None, temperature=0.7, max_tokens=3000, max_retries=3):
    """Call GPT API to generate questions with retry mechanism"""
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
    
    for attempt in range(max_retries):
        try:
            response = requests.post(API_BASE_URL, headers=headers, json=data, timeout=30)
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content'].strip()
            else:
                print(f"API error: {response.status_code}")
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) + random.uniform(0, 1)
                    print(f"Waiting {wait_time:.1f}s before retry (attempt {attempt + 1}/{max_retries})...")
                    time.sleep(wait_time)
                else:
                    print(f"Maximum retries reached, skipping request")
                    return None
        except Exception as e:
            print(f"API call failed: {e}")
            if attempt < max_retries - 1:
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                print(f"Waiting {wait_time:.1f}s before retry (attempt {attempt + 1}/{max_retries})...")
                time.sleep(wait_time)
            else:
                print(f"Maximum retries reached, skipping request")
                return None
    
    return None

def query_wikidata_sparql(query):
    """Execute SPARQL query"""
    endpoint = "https://query.wikidata.org/sparql"
    headers = {
        'User-Agent': 'Level1-Generator/1.0',
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

def verify_uniqueness(sparql_query):
    """Verify uniqueness of SPARQL query results"""
    result = query_wikidata_sparql(sparql_query)
    if result and result['results']['bindings']:
        if 'count' in result['results']['bindings'][0]:
            count = int(result['results']['bindings'][0]['count']['value'])
            return count == 1
        else:
            return len(result['results']['bindings']) == 1
    return False

def prepare_candidates_by_relation(df, total_candidates=2000):
    """Group by relation, collect candidates from each relation type"""
    print(f"Preparing {total_candidates} candidate triples...")
    
    # Group by relation
    relation_groups = defaultdict(list)
    for _, row in df.iterrows():
        relation_groups[row['predicate_id']].append(row)
    
    # Sort relations by triple count, prioritize relations with more triples
    sorted_relations = sorted(relation_groups.items(), key=lambda x: len(x[1]), reverse=True)
    print(f"Found {len(sorted_relations)} different relation types")
    
    candidates = []
    used_entities = set()
    
    # Calculate how many samples per relation
    samples_per_relation = max(1, total_candidates // len(sorted_relations))
    print(f"Expected {samples_per_relation} samples per relation")
    
    for relation_id, rows in sorted_relations:
        relation_label = rows[0]['predicate_label']
        print(f"Processing relation: {relation_label} ({len(rows)} triples)")
        
        # Find locally unique triples for this relation
        entity_relation_counts = defaultdict(set)
        for row in rows:
            key = (row['subject_id'], row['predicate_id'])
            entity_relation_counts[key].add(row['object_id'])
        
        # Select locally unique and unused entities
        relation_candidates = []
        for (subject_id, predicate_id), objects in entity_relation_counts.items():
            if len(objects) == 1 and subject_id not in used_entities:
                row = next(r for r in rows if r['subject_id'] == subject_id and r['predicate_id'] == predicate_id)
                relation_candidates.append(row)
        
        # Randomly select samples from this relation
        take_count = min(samples_per_relation, len(relation_candidates))
        selected = random.sample(relation_candidates, take_count) if relation_candidates else []
        
        for row in selected:
            candidates.append({
                'subject_id': row['subject_id'],
                'subject_label': row['subject_label'],
                'predicate_id': row['predicate_id'],
                'predicate_label': row['predicate_label'],
                'object_id': row['object_id'],
                'object_label': row['object_label']
            })
            used_entities.add(row['subject_id'])
            used_entities.add(row['object_id'])
        
        print(f"  -> Selected {len(selected)} candidates")
        
        if len(candidates) >= total_candidates:
            break
    
    # If not enough, supplement from remaining relations
    if len(candidates) < total_candidates:
        print(f"Current {len(candidates)} candidates, continuing collection...")
        remaining = total_candidates - len(candidates)
        for relation_id, rows in sorted_relations:
            if len(candidates) >= total_candidates:
                break
            # Continue taking more samples from this relation...
    
    print(f"Total prepared {len(candidates)} candidate triples, involving {len(used_entities)} different entities")
    return candidates[:total_candidates]

def verify_candidates_until_target(candidates, target_count=300):
    """Verify candidates one by one until target unique answers found"""
    print(f"Starting candidate verification, target {target_count} unique answers...")
    
    verified_triples = []
    
    for idx, candidate in enumerate(candidates):
        print(f"Verification progress: {idx+1}/{len(candidates)} (found {len(verified_triples)}/{target_count})")
        
        # Construct SPARQL query
        sparql_query = f"""
SELECT (COUNT(?object) AS ?count) WHERE {{
  wd:{candidate['subject_id']} wdt:{candidate['predicate_id']} ?object .
}}
"""
        
        # Verify uniqueness
        if verify_uniqueness(sparql_query):
            verified_triples.append({
                'subject_id': candidate['subject_id'],
                'subject_label': candidate['subject_label'],
                'predicate_id': candidate['predicate_id'],
                'predicate_label': candidate['predicate_label'],
                'object_id': candidate['object_id'],
                'object_label': candidate['object_label'],
                'sparql_query': sparql_query.strip(),
                'verified': True
            })
            
            print(f"  ✓ Verification passed: {candidate['subject_label']} -> {candidate['object_label']}")
            
            # Stop when target reached
            if len(verified_triples) >= target_count:
                print(f"Found {target_count} verified triples, stopping verification")
                break
        else:
            print(f"  ✗ Verification failed: {candidate['subject_label']} -> {candidate['object_label']}")
        
        time.sleep(0.1)  # Control query frequency
    
    print(f"Verification complete, found {len(verified_triples)} unique answer triples")
    return verified_triples

def generate_questions(unique_triples):
    """Use GPT to generate Level 1 questions"""
    print(f"Using GPT to generate {len(unique_triples)} Level 1 questions...")
    
    qa_pairs = []
    failed_count = 0
    
    for idx, triple in enumerate(unique_triples):
        if idx % 20 == 0:
            success_rate = len(qa_pairs) / max(1, idx) * 100 if idx > 0 else 0
            print(f"Question generation progress: {idx}/{len(unique_triples)} ({idx/len(unique_triples)*100:.1f}%)")
            print(f"  Successfully generated: {len(qa_pairs)}, failed: {failed_count}, success rate: {success_rate:.1f}%")
            if qa_pairs:
                recent_qa = qa_pairs[-1]
                print(f"  Latest example: Q: {recent_qa['question'][:60]}...")
                print(f"                  A: {recent_qa['answer']}")
        
        # Construct GPT prompt
        prompt = f"""
Generate a natural, clear question based on this knowledge triple:

Subject: {triple['subject_label']}
Relation: {triple['predicate_label']}
Answer: {triple['object_label']}

Requirements:
1. Make it conversational and natural
2. The answer should be exactly "{triple['object_label']}"
3. Ask about the relationship in a human-friendly way
4. Keep it concise and clear

Examples:
- "What is the capital of France?" → "Paris"
- "Who directed Inception?" → "Christopher Nolan"
- "Where was Einstein born?" → "Germany"

Generate only the question:"""

        messages = [
            {"role": "system", "content": "You generate natural questions from knowledge triples."},
            {"role": "user", "content": prompt}
        ]
        
        question = call_gpt_api(messages, temperature=0.7, max_tokens=3000, max_retries=5)
        
        if question and len(question.strip()) > 5:
            qa_pairs.append({
                'question': question.strip(),
                'answer': triple['object_label'],
                'level': 1,
                'type': 'single_hop',
                'reasoning_chain': [
                    [triple['subject_label'], triple['predicate_label'], triple['object_label']]
                ],
                'sparql_verification': triple['sparql_query'],
                'verified_with_sparql': True,
                'source_triple': {
                    'subject_id': triple['subject_id'],
                    'subject_label': triple['subject_label'],
                    'predicate_id': triple['predicate_id'],
                    'predicate_label': triple['predicate_label'],
                    'object_id': triple['object_id'],
                    'object_label': triple['object_label']
                }
            })
        else:
            failed_count += 1
            print(f"  ✗ Question {idx+1} generation failed: {triple['subject_label']} -> {triple['object_label']}")
        
        time.sleep(0.5)  # Increase delay to avoid rate limiting
    
    print(f"\nFinal statistics: Successfully generated {len(qa_pairs)} questions, failed {failed_count}")
    return qa_pairs

def main():
    """Main function: Generate 300 Level 1 questions"""
    
    # Load recent changed triple data
    print("Loading triple data...")
    df = pd.read_csv('./data/final_changed_item_with_id.csv')
    print(f"Loaded {len(df)} triples")
    
    # Use complete dataset to ensure relation diversity
    # Sample if data is too large
    if len(df) > 30000:
        df = df.sample(n=30000, random_state=42)
        print(f"Large dataset, sampled to {len(df)} triples for performance")
    
    # Step 1: Prepare candidate triples (grouped by relation, total 5000)
    candidates = prepare_candidates_by_relation(df, total_candidates=5000)
    
    # Step 2: Verify candidates until 600 unique answers found
    unique_triples = verify_candidates_until_target(candidates, target_count=600)
    
    # Generate questions
    level1_qa_pairs = generate_questions(unique_triples)
    
    # Save results
    result = {
        'metadata': {
            'description': 'Level 1 single-hop questions with unique answers',
            'total_questions': len(level1_qa_pairs),
            'verification_method': 'Mixed: SPARQL + local uniqueness',
            'sparql_verified_count': len(level1_qa_pairs),  # All questions are verified
            'generation_date': '2025-09-07'
        },
        'qa_pairs': level1_qa_pairs
    }
    
    # Save file
    filename = './outputs/level1_300_questions_2021.json'
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    # Display statistics and examples
    print(f"\n=== Generation Complete ===")
    print(f"Total generated: {len(level1_qa_pairs)} Level 1 questions")
    print(f"SPARQL verified: {result['metadata']['sparql_verified_count']} questions")
    print(f"Local uniqueness: {len(level1_qa_pairs) - result['metadata']['sparql_verified_count']} questions")
    print(f"Saved to: {filename}")
    
    # Display examples
    print(f"\n=== Question Examples ===")
    for i, qa in enumerate(level1_qa_pairs[:5]):
        verification_status = "✓SPARQL verified" if qa['verified_with_sparql'] else "○Local unique"
        print(f"{i+1}. [{verification_status}] Q: {qa['question']}")
        print(f"   A: {qa['answer']}")
        print(f"   Relation: {qa['source_triple']['predicate_label']}")
        print()
    
    # Statistics by relation type
    relation_counts = defaultdict(int)
    for qa in level1_qa_pairs:
        relation_counts[qa['source_triple']['predicate_label']] += 1
    
    print(f"\n=== Relation Type Distribution (Top 10) ===")
    sorted_relations = sorted(relation_counts.items(), key=lambda x: x[1], reverse=True)
    for relation, count in sorted_relations[:10]:
        print(f"{relation}: {count} questions")

if __name__ == "__main__":
    main()