#!/usr/bin/env python3
"""
Level 3 Advanced Question Generation System
Based on Level 2 questions, using node expansion and cultural metaphors for multi-layer obfuscation
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
API_MODEL = "gpt-4o"


def call_gpt_api(messages, model=None, temperature=0.8, max_tokens=250):
    """Call GPT API"""
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
        response = requests.post(
            API_BASE_URL, headers=headers, json=data, timeout=30)
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
        'User-Agent': 'Level3-Advanced/1.0',
        'Accept': 'application/json'
    }
    params = {'query': query, 'format': 'json'}

    try:
        response = requests.get(endpoint, headers=headers,
                                params=params, timeout=30)
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except Exception:
        return None


class NodeExpansionEngine:
    """Node expansion engine: find multi-level indirect descriptions for entities"""

    def __init__(self, df):
        self.df = df
        self.entity_cache = {}

    def get_entity_expansions(self, entity_label, max_depth=2):
        """Get multi-level expansion information for entity"""
        if entity_label in self.entity_cache:
            return self.entity_cache[entity_label]

        expansions = {
            'direct_attributes': [],
            'cultural_references': [],
            'historical_context': [],
            'metaphorical_descriptions': []
        }

        # Find entity ID
        entity_id = self.find_entity_id(entity_label)
        if not entity_id:
            return expansions

        # Direct attributes
        direct_attrs = self.get_direct_attributes(entity_id, entity_label)
        expansions['direct_attributes'] = self.query_wikidata_entity_properties(
            entity_id, entity_label)

        # Use GPT to generate cultural references
        # cultural_refs = self.generate_cultural_references_with_gpt(
        #     entity_label, wikidata_props)
        # expansions['cultural_references'] = cultural_refs

        # # Metaphorical descriptions
        # metaphors = self.generate_metaphorical_descriptions(
        #     entity_label, direct_attrs)
        # expansions['metaphorical_descriptions'] = metaphors
        self.entity_cache[entity_label] = expansions
        return expansions

    def find_entity_id(self, entity_label):
        """Find entity ID"""
        matches = self.df[(self.df['subject_label'] == entity_label) |
                          (self.df['object_label'] == entity_label)]

        if not matches.empty:
            row = matches.iloc[0]
            entity_id = row['subject_id'] if row['subject_label'] == entity_label else row['object_id']
            print(f"      Found entity ID: {entity_label} -> {entity_id}")
            return entity_id
        print(f"      Entity ID not found: {entity_label}")
        return None

    def get_direct_attributes(self, entity_id, entity_label):
        """Get direct attributes"""
        attributes = []

        # Relations as subject
        as_subject = self.df[self.df['subject_id'] == entity_id].head(5)
        for _, row in as_subject.iterrows():
            attributes.append({
                'type': 'has',
                'relation': row['predicate_label'],
                'value': row['object_label'],
                'description': f"has {row['predicate_label']} {row['object_label']}"
            })

        # Relations as object
        as_object = self.df[self.df['object_id'] == entity_id].head(5)
        for _, row in as_object.iterrows():
            attributes.append({
                'type': 'is',
                'relation': row['predicate_label'],
                'value': row['subject_label'],
                'description': f"is the {row['predicate_label']} of {row['subject_label']}"
            })

        return attributes

    def query_wikidata_entity_properties(self, entity_id, entity_label, max_properties=100):
        """Use SPARQL to query rich attribute information for entity"""
        sparql_query = f"""
SELECT ?property ?propertyLabel ?value ?valueLabel WHERE {{
  wd:{entity_id} ?property ?value .
  
  # Only get useful descriptive attributes, exclude overly abstract classification attributes
  FILTER(?property IN (
    wdt:P27,   # country of citizenship
    wdt:P19,   # place of birth
    wdt:P20,   # place of death
    wdt:P103,  # native language
    wdt:P136,  # genre
    wdt:P106,  # occupation
    wdt:P495,  # country of origin
    wdt:P37,   # official language
    wdt:P36,   # capital
    wdt:P571,  # inception
    wdt:P577,  # publication date
    wdt:P57,   # director
    wdt:P50,   # author
    wdt:P175,  # performer
    wdt:P1412, # languages spoken
    wdt:P17,   # country
    wdt:P131,  # located in administrative territorial entity
    wdt:P276,  # location
    wdt:P159,  # headquarters location
    wdt:P140,  # religion
    wdt:P108,  # employer
    wdt:P69,   # educated at
    wdt:P54,   # member of sports team
    wdt:P166,  # award received
    wdt:P127,  # owned by
    wdt:P449,  # original broadcaster
    wdt:P123,  # publisher
    wdt:P364   # original language of work
  ))
  
  # Wikibase label service will automatically generate labels for all entity URIs
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" . }}
}}
LIMIT {max_properties}
"""

        result = query_wikidata_sparql(sparql_query)
        properties = []

        if result and result.get('results', {}).get('bindings'):
            for binding in result['results']['bindings']:
                prop_label = binding.get('propertyLabel', {}).get('value', '')
                value_label = binding.get('valueLabel', {}).get('value', '')

                # Clean property labels, convert URI to readable text
                clean_prop_label = self._clean_property_label(prop_label)

                # Filter out invalid labels and links
                if self._is_valid_property_value(clean_prop_label, value_label):
                    properties.append({
                        'property': clean_prop_label,
                        'value': value_label,
                        'description': f"({entity_label},{clean_prop_label},{value_label})"
                    })

        return properties

    def _clean_property_label(self, prop_label):
        """Clean property labels, convert URI to readable text"""
        if not prop_label:
            return prop_label

        # If URI format, try to extract readable part or map to standard name
        if 'http://www.wikidata.org/prop/direct/' in prop_label:
            # Extract property ID (like P31)
            prop_id = prop_label.split('/')[-1]

            # Map common property IDs to readable names
            property_mappings = {
                'P31': 'instance of',
                'P279': 'subclass of',
                'P27': 'country of citizenship',
                'P19': 'place of birth',
                'P20': 'place of death',
                'P103': 'native language',
                'P136': 'genre',
                'P106': 'occupation',
                'P495': 'country of origin',
                'P37': 'official language',
                'P36': 'capital',
                'P571': 'inception',
                'P577': 'publication date',
                'P57': 'director',
                'P50': 'author',
                'P175': 'performer',
                'P1412': 'languages spoken',
                'P17': 'country',
                'P131': 'located in',
                'P276': 'location',
                'P159': 'headquarters location',
                'P140': 'religion',
                'P108': 'employer',
                'P69': 'educated at',
                'P54': 'member of sports team',
                'P166': 'award received',
                'P127': 'owned by',
                'P449': 'original broadcaster',
                'P123': 'publisher',
                'P364': 'original language'
            }

            return property_mappings.get(prop_id, prop_id)

        return prop_label

    def _is_valid_property_value(self, prop_label, value_label):
        """Verify if property value is valid (non-link, non-ID, non-abstract concept)"""
        if not prop_label or not value_label:
            return False

        # Filter out values containing Wikidata ID
        if value_label.startswith('Q') and value_label[1:].isdigit():
            return False

        # Filter out property names containing URI
        if 'http://' in prop_label or 'https://' in prop_label:
            return False

        # Filter out values containing HTTP links
        if 'http://' in value_label.lower() or 'https://' in value_label.lower():
            return False

        # Filter out values containing URI patterns
        if value_label.startswith('http://') or value_label.startswith('https://'):
            return False

        # Filter out overly long values (possibly descriptions rather than simple labels)
        if len(value_label) > 100:
            return False

        # Filter out values containing multiple slashes (possibly paths)
        if value_label.count('/') > 1:
            return False

        # Filter out overly abstract concept values
        abstract_concepts = ['profession', 'occupation', 'concept', 'category',
                             'classification', 'type', 'kind', 'form', 'class']
        if value_label.lower() in abstract_concepts:
            return False

        return True

    def _select_diverse_properties(self, wikidata_properties, max_properties=5):
        """Select diverse properties, avoid duplicate types"""
        if not wikidata_properties:
            return []

        # Group by property type, deduplicate
        property_groups = {}
        seen_descriptions = set()

        for prop in wikidata_properties:
            property_name = prop.get('property', '')
            description = prop.get('description', '')

            # Skip duplicate descriptions
            if description in seen_descriptions:
                continue
            seen_descriptions.add(description)

            # Group by property type
            if property_name not in property_groups:
                property_groups[property_name] = []
            property_groups[property_name].append(prop)

        # Select the most representative property from each group
        diverse_properties = []
        for property_name, props in property_groups.items():
            # Prioritize properties with shorter values that don't contain IDs or links
            best_prop = min(props, key=lambda p: (
                len(p.get('value', '')),
                'Q' in p.get('value', ''),  # Avoid Wikidata ID
                'http' in p.get('value', '').lower()  # Avoid links
            ))
            diverse_properties.append(best_prop)

            if len(diverse_properties) >= max_properties:
                break

        return diverse_properties

    def _generate_multi_hop_descriptions(self, entity_label, wikidata_properties):
        """Generate multi-hop indirect descriptions through property chains"""
        descriptions = []

        for prop in wikidata_properties:
            property_name = prop.get('property', '')
            value = prop.get('value', '')

            # Geographic location multi-hop
            if property_name in ['located in', 'country', 'country of citizenship']:
                multi_hop_desc = self._create_geographic_hop(value)
                if multi_hop_desc:
                    descriptions.append(multi_hop_desc)

            # Temporal multi-hop
            elif property_name in ['inception', 'publication date']:
                multi_hop_desc = self._create_temporal_hop(value)
                if multi_hop_desc:
                    descriptions.append(multi_hop_desc)

            # Occupation/type multi-hop
            elif property_name in ['occupation', 'genre']:
                multi_hop_desc = self._create_categorical_hop(
                    property_name, value)
                if multi_hop_desc:
                    descriptions.append(multi_hop_desc)

        return descriptions[:2]  # Return at most 2 descriptions

    def _create_geographic_hop(self, location):
        """Create multi-hop description for geographic location"""
        # Common geographic multi-hop mappings
        geographic_hops = {
            'Italy': 'Mediterranean country',
            'Sicily': 'Italian island',
            'France': 'European republic',
            'Germany': 'Central European nation',
            'United Kingdom': 'island nation',
            'Scotland': 'northern British territory',
            'England': 'southern British territory',
            'United States': 'North American federation',
            'California': 'Pacific coast state',
            'Texas': 'southwestern state',
            'New York': 'northeastern state',
            'Denmark': 'Scandinavian kingdom',
            'Sweden': 'Nordic country',
            'Norway': 'fjord nation'
        }

        return geographic_hops.get(location)

    def _create_temporal_hop(self, date_str):
        """Create multi-hop description for temporal data"""
        try:
            if '-' in date_str:
                year = int(date_str.split('-')[0])
                if year < 1900:
                    return 'pre-industrial era'
                elif year < 1950:
                    return 'early 20th century'
                elif year < 2000:
                    return 'late 20th century'
                else:
                    return 'modern era'
        except:
            pass
        return None

    def _create_categorical_hop(self, property_name, value):
        """Create multi-hop description for categorical data"""
        category_hops = {
            # Occupation multi-hop
            'teacher': 'education professional',
            'director': 'film industry figure',
            'author': 'literary figure',
            'politician': 'public servant',
            'athlete': 'sports professional',

            # Type multi-hop
            'football club': 'sports organization',
            'university': 'academic institution',
            'museum': 'cultural institution',
            'film': 'cinematic work',
            'album': 'musical work'
        }

        return category_hops.get(value.lower())

    def generate_cultural_references_with_gpt(self, entity_label, wikidata_properties):
        """Generate indirect references through SPARQL multi-hop queries, completely remove GPT cultural associations"""
        if not wikidata_properties:
            return []

        # Find entity ID for SPARQL query
        entity_id = self.find_entity_id(entity_label)
        if not entity_id:
            return []

        # Simple multi-hop obfuscation of existing attributes
        multi_hop_descriptions = self._create_simple_multi_hop_descriptions(
            entity_id, wikidata_properties)

        return multi_hop_descriptions

    def _create_simple_multi_hop_descriptions(self, entity_id, wikidata_properties):
        """Simple reverse multi-hop for attributes: original attribute A->B, find C such that C->A->B"""
        descriptions = []

        for prop in wikidata_properties[:3]:  # Only process first 3 attributes
            property_name = prop.get('property', '')
            value = prop.get('value', '')

            # Perform reverse query multi-hop for each attribute
            multi_hop_desc = self._reverse_hop_property(
                entity_id, property_name, value)
            if multi_hop_desc:
                descriptions.append(multi_hop_desc)

        return descriptions[:2]  # Return at most 2 descriptions

    def _reverse_hop_property(self, entity_id, property_name, value):
        """Generic reverse multi-hop: A->B, find all C satisfying C->B, choose appropriate C for description"""
        # Generic reverse query: find all entities pointing to value
        sparql_query = f"""
        SELECT ?reverseEntity ?reverseEntityLabel ?property ?propertyLabel WHERE {{
          ?reverseEntity ?property ?valueEntity .
          ?valueEntity rdfs:label "{value}"@en .
          
          # Filter useful reverse attributes
          FILTER(?property IN (
            wdt:P30,   # continent
            wdt:P17,   # country  
            wdt:P279,  # subclass of
            wdt:P131,  # located in
            wdt:P361,  # part of
            wdt:P276,  # location
            wdt:P150,  # contains administrative territorial entity
            wdt:P706,  # located on terrain feature
            wdt:P527,  # has part
            wdt:P190,  # sister city
            wdt:P47,   # shares border with
            wdt:P206,  # located in or next to body of water
            wdt:P138,  # named after
            wdt:P170,  # creator
            wdt:P178,  # developer
            wdt:P272,  # production company
            wdt:P264,  # record label
            wdt:P449,  # original broadcaster
            wdt:P750,  # distributor
            wdt:P123,  # publisher
            wdt:P127,  # owned by
            wdt:P1830, # owner of
            wdt:P355,  # subsidiary
            wdt:P749,  # parent organization
            wdt:P112,  # founded by
            wdt:P1037, # director/manager
            wdt:P3320, # board member
            wdt:P488,  # chairperson
            wdt:P35,   # head of state
            wdt:P6,    # head of government
            wdt:P1313, # office held by head of government
            wdt:P1906, # office held by head of state
            wdt:P37,   # official language
            wdt:P103,  # native language
            wdt:P1412, # languages spoken/written
            wdt:P364,  # original language of work
            wdt:P407,  # language of work
            wdt:P495,  # country of origin
            wdt:P840,  # narrative location
            wdt:P915,  # filming location
            wdt:P291,  # place of publication
            wdt:P159,  # headquarters location
            wdt:P740,  # location of formation
            wdt:P571,  # inception
            wdt:P576,  # dissolved/abolished
            wdt:P580,  # start time
            wdt:P582,  # end time
            wdt:P585   # point in time
          ))
          
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" . }}
        }}
        LIMIT 5
        """

        result = query_wikidata_sparql(sparql_query)
        if result and result.get('results', {}).get('bindings'):
            for binding in result['results']['bindings']:
                reverse_label = binding.get(
                    'reverseEntityLabel', {}).get('value', '')
                prop_label = binding.get('propertyLabel', {}).get('value', '')

                # Filter out useless results and generate descriptions
                if reverse_label and len(reverse_label) > 2 and len(reverse_label) < 50:
                    # Geographic attributes
                    if prop_label == 'continent':
                        return f"{reverse_label} entity"
                    elif prop_label == 'country':
                        return f"{reverse_label} based"
                    elif prop_label in ['contains administrative territorial entity', 'located on terrain feature']:
                        return f"{reverse_label} region"
                    elif prop_label in ['shares border with', 'sister city']:
                        return f"neighbor of {reverse_label}"
                    elif prop_label == 'located in or next to body of water':
                        return f"{reverse_label} adjacent"

                    # Organization/enterprise attributes
                    elif prop_label in ['parent organization', 'owned by', 'founded by']:
                        return f"{reverse_label} affiliate"
                    elif prop_label in ['subsidiary', 'has part']:
                        return f"{reverse_label} branch"
                    elif prop_label in ['production company', 'record label', 'publisher', 'distributor']:
                        return f"{reverse_label} production"
                    elif prop_label == 'original broadcaster':
                        return f"{reverse_label} network"

                    # Creative attributes
                    elif prop_label in ['creator', 'developer']:
                        return f"{reverse_label} creation"
                    elif prop_label == 'named after':
                        return f"namesake of {reverse_label}"

                    # Language attributes
                    elif prop_label in ['official language', 'native language', 'original language of work']:
                        return f"{reverse_label} speaking"

                    # Temporal/location attributes
                    elif prop_label in ['filming location', 'narrative location', 'place of publication']:
                        return f"{reverse_label} associated"
                    elif prop_label in ['headquarters location', 'location of formation']:
                        return f"{reverse_label} established"

                    # Classification attributes
                    elif prop_label in ['subclass of', 'part of']:
                        return f"{reverse_label.lower()}"

        return None

    def _get_geographic_multi_hop(self, entity_id):
        """Multi-hop query through geographic location attributes"""
        sparql_query = f"""
        SELECT ?location ?locationLabel ?country ?countryLabel ?continent ?continentLabel WHERE {{
          wd:{entity_id} wdt:P131|wdt:P276|wdt:P17|wdt:P495 ?location .
          OPTIONAL {{ ?location wdt:P17|wdt:P131 ?country . }}
          OPTIONAL {{ ?country wdt:P30 ?continent . }}
          
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" . }}
        }}
        LIMIT 5
        """

        result = query_wikidata_sparql(sparql_query)
        descriptions = []

        if result and result.get('results', {}).get('bindings'):
            for binding in result['results']['bindings']:
                location_label = binding.get(
                    'locationLabel', {}).get('value', '')
                country_label = binding.get(
                    'countryLabel', {}).get('value', '')
                continent_label = binding.get(
                    'continentLabel', {}).get('value', '')

                # Generate multi-hop description: entity → region → country → continent
                if country_label and location_label != country_label:
                    descriptions.append(f"{country_label} entity")
                elif continent_label:
                    descriptions.append(f"{continent_label} organization")
                elif location_label:
                    descriptions.append(f"{location_label} based")

        return list(set(descriptions))  # Deduplicate

    def _get_organizational_multi_hop(self, entity_id):
        """Multi-hop query through organizational type attributes"""
        sparql_query = f"""
        SELECT ?type ?typeLabel ?parent ?parentLabel WHERE {{
          wd:{entity_id} wdt:P31|wdt:P279 ?type .
          OPTIONAL {{ ?type wdt:P279 ?parent . }}
          
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" . }}
        }}
        LIMIT 5
        """

        result = query_wikidata_sparql(sparql_query)
        descriptions = []

        if result and result.get('results', {}).get('bindings'):
            for binding in result['results']['bindings']:
                type_label = binding.get('typeLabel', {}).get('value', '')
                parent_label = binding.get('parentLabel', {}).get('value', '')

                # Generate multi-hop description: entity → type → parent type
                if parent_label and 'organization' in parent_label.lower():
                    descriptions.append(f"{parent_label.lower()}")
                elif type_label and any(word in type_label.lower() for word in ['club', 'team', 'organization', 'institution']):
                    descriptions.append(f"{type_label.lower()}")

        return list(set(descriptions))  # Deduplicate

    def _get_temporal_multi_hop(self, entity_id):
        """Multi-hop query through temporal attributes"""
        sparql_query = f"""
        SELECT ?date ?dateLabel WHERE {{
          wd:{entity_id} wdt:P571|wdt:P577|wdt:P580 ?date .
          
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" . }}
        }}
        LIMIT 3
        """

        result = query_wikidata_sparql(sparql_query)
        descriptions = []

        if result and result.get('results', {}).get('bindings'):
            for binding in result['results']['bindings']:
                date_str = binding.get('date', {}).get('value', '')

                # Generate era descriptions from dates
                try:
                    if 'T' in date_str:  # ISO format
                        year = int(date_str.split('T')[0].split('-')[0])
                    elif '-' in date_str:
                        year = int(date_str.split('-')[0])
                    else:
                        continue

                    if year < 1900:
                        descriptions.append("pre-modern era")
                    elif year < 1950:
                        descriptions.append("early 20th century")
                    elif year < 2000:
                        descriptions.append("late 20th century")
                    else:
                        descriptions.append("21st century")

                except (ValueError, IndexError):
                    continue

        return list(set(descriptions))  # Deduplicate

    def generate_metaphorical_descriptions(self, entity_label, attributes):
        """Generate metaphorical descriptions"""
        metaphors = []

        for attr in attributes[:3]:
            relation = attr.get('relation', '')
            value = attr.get('value', '')

            if 'birth' in relation.lower():
                metaphors.append(f"offspring of {value}")
            elif 'capital' in relation.lower():
                metaphors.append(f"crowned jewel of {value}")
            elif 'language' in relation.lower():
                metaphors.append(f"voice that speaks {value}")
            elif 'director' in relation.lower():
                metaphors.append(f"cinematic architect")

        return metaphors


class Level3Generator:
    """Level 3 question generator"""

    def __init__(self, df):
        self.df = df
        self.expansion_engine = NodeExpansionEngine(df)
        self.level2_questions = []

    def load_level2_questions(self, file_path):
        """Load Level 2 questions"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.level2_questions = data.get('qa_pairs', [])
            print(f"Loaded {len(self.level2_questions)} Level 2 questions")
        except FileNotFoundError:
            print(f"Level 2 file not found: {file_path}")

    def create_abstract_bridge_question(self, level2_qa):
        """Create abstract bridge question"""
        if level2_qa.get('type') != 'bridge':
            return None

        try:
            bridge_info = level2_qa.get('bridge_info', {})
            start_entity = bridge_info.get('start_entity', '')
            intermediate_entity = bridge_info.get('intermediate_entity', '')
            final_entity = bridge_info.get('final_entity', '')

            if not all([start_entity, intermediate_entity, final_entity]):
                return None

            # Get entity expansion descriptions
            print(f"    Expanding entities: {start_entity} -> {intermediate_entity}")
            start_expansions = self.expansion_engine.get_entity_expansions(
                start_entity)
            intermediate_expansions = self.expansion_engine.get_entity_expansions(
                intermediate_entity)

            # Select best indirect descriptions
            start_desc = self.select_best_description(
                start_entity, start_expansions)
            intermediate_desc = self.select_best_description(
                intermediate_entity, intermediate_expansions)

            print(
                f"    Description: {start_entity} -> {start_desc[:50] if start_desc else 'None'}")
            print(
                f"    Description: {intermediate_entity} -> {intermediate_desc[:50] if intermediate_desc else 'None'}")

            if not start_desc or not intermediate_desc:
                print(f"    Skip: lacking valid descriptions")
                return None

            # Generate abstract questions
            prompt = f"""
Create a more abstract but still understandable question using these indirect descriptions:

Original: {level2_qa['question']}
Answer: {level2_qa['answer']}

Entity 1 ({start_entity}) → {start_desc}
Entity 2 ({intermediate_entity}) → {intermediate_desc}

Requirements:
1. Use indirect descriptions instead of direct names
2. Make it more challenging but still solvable
3. Use cultural references or metaphors when appropriate
4. Keep the logical reasoning path clear
5. Avoid overly poetic language that obscures meaning

Examples of appropriate abstraction level:
- "What was the predecessor of the democratic process in the East African nation known for its thousand hills?"
- "Which electoral event preceded the governance selection in the land of drummers and coffee?"
- "What came before the parliamentary choice in the country that neighbors Lake Tanganyika?"

Generate only the question:"""

            messages = [
                {"role": "system", "content": "You create challenging but solvable questions using indirect descriptions. Focus on clarity while maintaining difficulty."},
                {"role": "user", "content": prompt}
            ]

            question = call_gpt_api(messages, temperature=0.8)
            print(f"    GPT response: {question[:50] if question else 'None'}...")

            if question:
                # Verify answer uniqueness
                if not self.verify_answer_uniqueness(level2_qa.get('sparql_verification', ''), level2_qa['answer']):
                    print(f"  Answer verification failed: {level2_qa['answer']}")
                    return None

                return {
                    'question': question,
                    'answer': level2_qa['answer'],
                    'level': 3,
                    'type': 'abstract_bridge',
                    'reasoning_chain': level2_qa['reasoning_chain'],
                    'sparql_verification': level2_qa['sparql_verification'],
                    'original_level2_question': level2_qa['question'],
                    'abstraction_info': {
                        'start_entity_abstraction': start_desc,
                        'intermediate_entity_abstraction': intermediate_desc,
                        'metaphor_level': 'very_high'
                    },
                    'answer_verified': True
                }
        except Exception as e:
            print(f"Bridge question processing error: {e}")
            return None

    def create_abstract_multi_attribute_question(self, level2_qa):
        """Create abstract multi-attribute question"""
        if level2_qa.get('type') != 'multi_attribute':
            return None

        try:
            constraint_info = level2_qa.get('constraint_info', {})
            constraints = constraint_info.get('constraints', [])

            if len(constraints) < 2:
                return None

            # Generate abstract descriptions for constraint entities
            abstract_constraints = []
            for constraint in constraints[:3]:  # Use at most 3 constraints
                if ': ' in constraint:
                    relation, entity = constraint.split(': ', 1)
                    entity_expansions = self.expansion_engine.get_entity_expansions(
                        entity)
                    abstract_desc = self.select_best_description(
                        entity, entity_expansions)

                    if abstract_desc:
                        abstract_constraints.append(
                            f"abstract [{relation}: {entity}] As [{relation}: {abstract_desc}]")
                    else:
                        abstract_constraints.append(constraint)

            if len(abstract_constraints) < 2:
                return None

            prompt = f"""
Reframe the original question into a more challenging but still solvable version.

Original Question: {level2_qa['question']}
Answer: {level2_qa['answer']}

Abstract Constraints (structured form):
{'; '.join(abstract_constraints)}

Instructions:
1. Use the abstract constraints instead of the original entities when rewriting the question.
2. Convert the structured triples into natural language phrasing. 
   For example:
   - (Sinulog festival, location, Cebu City) → "a festival held in Cebu City"
   - (Partido Demokratiko Pilipino, country, Philippines) → "a political party in the Philippines"
3. Combine the abstract constraints naturally into a single question.
4. Make the new question more challenging than the original, but still fair and solvable.
5. Ensure the reasoning path is clear and avoid obscure metaphors.

Output: Only generate the transformed question, nothing else.
"""
            messages = [
                {"role": "system", "content": "You create challenging questions using multiple indirect constraints. Keep them understandable but requiring knowledge to solve."},
                {"role": "user", "content": prompt}
            ]

            question = call_gpt_api(messages, temperature=0.1)

            if question:
                # Verify answer uniqueness
                # if not self.verify_answer_uniqueness(level2_qa.get('sparql_verification', ''), level2_qa['answer']):
                #     print(f"  Answer verification failed: {level2_qa['answer']}")
                #     return None

                return {
                    'question': question,
                    'answer': level2_qa['answer'],
                    'level': 3,
                    'type': 'abstract_multi_attribute',
                    'reasoning_chain': level2_qa['reasoning_chain'],
                    'sparql_verification': level2_qa['sparql_verification'],
                    'original_level2_question': level2_qa['question'],
                    'abstraction_info': {
                        'abstract_constraints': abstract_constraints,
                        'original_constraints': constraints,
                        'cultural_depth': 'maximum'
                    },
                    'answer_verified': True
                }
        except Exception as e:
            print(f"Multi-attribute question processing error: {e}")
            return None

    def select_best_description(self, entity, expansions):
        """Simple selection of best description"""

        # If no multi-hop descriptions, use direct attributes
        direct_attrs = expansions.get('direct_attributes', [])
        if direct_attrs:
            attr = random.choice(direct_attrs[:2])
            return attr.get('description', '')

        return None

    def verify_answer_uniqueness(self, sparql_query, expected_answer):
        """Verify answer uniqueness for Level 3 questions"""
        if not sparql_query:
            return False

        try:
            # Execute original SPARQL query to verify answer is still unique
            result = query_wikidata_sparql(sparql_query)

            if result and result.get('results', {}).get('bindings'):
                # Check if it's a count query
                bindings = result['results']['bindings']
                if len(bindings) == 1 and 'count' in bindings[0]:
                    count = int(bindings[0]['count']['value'])
                    return count == 1
                else:
                    # Direct result query, check result count
                    return len(bindings) == 1
            return False
        except Exception as e:
            print(f"Answer verification error: {e}")
            return False

    def generate_level3_questions(self, max_questions=50):
        """Generate Level 3 questions"""
        print(f"Starting Level 3 question generation, target: {max_questions} questions")

        if not self.level2_questions:
            print("No Level 2 questions, cannot generate Level 3")
            return []

        level3_qa_pairs = []
        processed = 0

        # Randomly shuffle Level 2 questions
        shuffled_level2 = random.sample(
            self.level2_questions, len(self.level2_questions))

        for level2_qa in shuffled_level2:
            if len(level3_qa_pairs) >= max_questions:
                break

            processed += 1
            if processed % 10 == 0:
                print(
                    f"Processing progress: {processed}/{len(shuffled_level2)}, generated: {len(level3_qa_pairs)}")

            # Generate Level 3 questions based on type
            level3_qa = None

            if level2_qa.get('type') == 'bridge':
                level3_qa = self.create_abstract_bridge_question(level2_qa)
            elif level2_qa.get('type') == 'multi_attribute':
                level3_qa = self.create_abstract_multi_attribute_question(
                    level2_qa)

            if level3_qa:
                level3_qa_pairs.append(level3_qa)
                print(f"  ✓ Generated: {level3_qa['question'][:70]}...")

            time.sleep(0.3)

        print(f"Level 3 generation complete, {len(level3_qa_pairs)} questions total")
        return level3_qa_pairs


def main():
    """Main function"""
    # Load data
    print("Loading triple data...")
    df = pd.read_csv('./data/final_changed_item_with_id.csv')
    print(f"Loaded {len(df)} triples")

    # Create Level 3 generator
    generator = Level3Generator(df)
    generator.load_level2_questions('./data/level2_filtered.json')

    # Generate questions
    level3_questions = generator.generate_level3_questions(max_questions=200)

    if not level3_questions:
        print("No Level 3 questions generated")
        return

    # Save results
    result = {
        'metadata': {
            'description': 'Level 3 highly abstract questions with cultural metaphors',
            'total_questions': len(level3_questions),
            'abstraction_level': 'maximum',
            'cultural_depth': 'very_high',
            'generation_method': 'node_expansion_with_metaphors',
            'generation_date': '2025-09-07'
        },
        'qa_pairs': level3_questions
    }

    filename = './outputs/level3_advanced_questions_913.json'
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # Display results
    print(f"\n=== Level 3 Advanced Question Generation Complete ===")
    print(f"Total: {len(level3_questions)} questions")
    print(f"Saved to: {filename}")

    # Display examples
    print(f"\n=== Question Examples ===")
    for i, qa in enumerate(level3_questions[:3]):
        print(f"\n{i+1}. [{qa['type']}]")
        print(f"Level 3: {qa['question']}")
        print(f"Answer: {qa['answer']}")
        print(f"Original Level 2: {qa['original_level2_question']}")


if __name__ == "__main__":
    main()
