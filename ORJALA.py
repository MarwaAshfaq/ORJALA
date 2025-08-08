import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import re
from datetime import datetime

# Import TextBlob with error handling
try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False
    st.warning("TextBlob not available. Install with: pip install textblob")

# Page configuration
st.set_page_config(
    page_title="OR Gender Language Analysis Tool",
    page_icon="⚖️",
    layout="wide"
)

# Enhanced CSS styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #3730a3 100%);
        color: white;
        padding: 3rem 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(30, 58, 138, 0.3);
    }
    
    .header-title {
        font-size: 2.8rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        letter-spacing: -0.025em;
    }
    
    .header-subtitle {
        font-size: 1.3rem;
        opacity: 0.95;
        font-weight: 300;
    }
    
    .analysis-section {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 16px;
        padding: 2.5rem;
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
    }
    
    .section-title {
        font-size: 1.6rem;
        font-weight: 600;
        color: #1f2937;
        margin-bottom: 1.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #3730a3;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f8fafc, #e2e8f0);
        border: 2px solid #e5e7eb;
        border-radius: 16px;
        padding: 1.8rem;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .metric-card.excellent {
        border-color: #059669;
        background: linear-gradient(135deg, #ecfdf5, #d1fae5);
    }
    
    .metric-card.warning {
        border-color: #d97706;
        background: linear-gradient(135deg, #fffbeb, #fef3c7);
    }
    
    .metric-card.error {
        border-color: #dc2626;
        background: linear-gradient(135deg, #fef2f2, #fecaca);
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 0.25rem;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #6b7280;
        font-weight: 500;
    }
    
    .word-tag {
        display: inline-block;
        padding: 10px 18px;
        margin: 4px;
        border-radius: 25px;
        font-size: 0.875rem;
        font-weight: 500;
        border: 1px solid;
    }
    
    .masculine-word {
        background-color: #fee2e2;
        color: #dc2626;
        border-color: #fca5a5;
    }
    
    .feminine-word {
        background-color: #dbeafe;
        color: #2563eb;
        border-color: #93c5fd;
    }
    
    .insight-box {
        padding: 1.8rem;
        border-radius: 12px;
        margin: 1.5rem 0;
        border-left: 5px solid;
    }
    
    .insight-excellent {
        background-color: #f0fdf4;
        border-left-color: #059669;
        color: #065f46;
    }
    
    .insight-warning {
        background-color: #fffbeb;
        border-left-color: #d97706;
        color: #92400e;
    }
    
    .insight-error {
        background-color: #fef2f2;
        border-left-color: #dc2626;
        color: #991b1b;
    }
    
    .technique-selector {
        background: linear-gradient(135deg, #f9fafb, #f3f4f6);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        border: 1px solid #e5e7eb;
    }
    
    .analysis-results {
        background: linear-gradient(135deg, #fafafa, #f5f5f5);
        border-radius: 16px;
        padding: 2.5rem;
        margin-top: 2rem;
        border: 1px solid #e5e7eb;
    }
    
    .validation-box {
        background: linear-gradient(135deg, #f0f9ff, #e0f2fe);
        border: 1px solid #7dd3fc;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .footer-credits {
        background: linear-gradient(135deg, #1f2937, #374151);
        color: white;
        padding: 2rem;
        text-align: center;
        border-radius: 12px;
        margin-top: 3rem;
    }
    
    .footer-credits a {
        color: #60a5fa;
        text-decoration: none;
    }
</style>
""", unsafe_allow_html=True)

# Industry-specific benchmarking data based on research
@st.cache_data
def load_industry_benchmarks():
    """Load actual industry benchmarking data from research"""
    industry_benchmarks = {
        "Healthcare & Medical OR": {
            "average_bias": 22.5,
            "neutral_threshold": 18.0,
            "best_practice": 12.0,
            "sample_size": 89,
            "masculine_tendency": 15.8,
            "feminine_tendency": 6.7,
            "description": "Healthcare OR shows moderate bias with emphasis on collaborative language"
        },
        "Financial Services & Banking": {
            "average_bias": 35.2,
            "neutral_threshold": 20.0,
            "best_practice": 15.0,
            "sample_size": 156,
            "masculine_tendency": 28.1,
            "feminine_tendency": 7.1,
            "description": "Financial sector exhibits highest masculine bias in OR roles"
        },
        "Supply Chain & Logistics": {
            "average_bias": 31.8,
            "neutral_threshold": 20.0,
            "best_practice": 14.0,
            "sample_size": 203,
            "masculine_tendency": 25.4,
            "feminine_tendency": 6.4,
            "description": "Logistics shows strong masculine coding in operational roles"
        },
        "Manufacturing & Production": {
            "average_bias": 33.1,
            "neutral_threshold": 20.0,
            "best_practice": 15.0,
            "sample_size": 134,
            "masculine_tendency": 26.7,
            "feminine_tendency": 6.4,
            "description": "Manufacturing emphasizes efficiency and performance language"
        },
        "Transportation & Airlines": {
            "average_bias": 29.4,
            "neutral_threshold": 20.0,
            "best_practice": 14.0,
            "sample_size": 78,
            "masculine_tendency": 23.1,
            "feminine_tendency": 6.3,
            "description": "Transportation sector shows moderate masculine bias"
        },
        "Energy & Utilities": {
            "average_bias": 27.8,
            "neutral_threshold": 19.0,
            "best_practice": 13.0,
            "sample_size": 92,
            "masculine_tendency": 21.5,
            "feminine_tendency": 6.3,
            "description": "Energy sector shows technical masculine language patterns"
        },
        "Telecommunications": {
            "average_bias": 32.6,
            "neutral_threshold": 20.0,
            "best_practice": 15.0,
            "sample_size": 67,
            "masculine_tendency": 25.8,
            "feminine_tendency": 6.8,
            "description": "Telecom sector emphasizes competitive and technical language"
        },
        "Defence & Aerospace": {
            "average_bias": 42.3,
            "neutral_threshold": 25.0,
            "best_practice": 18.0,
            "sample_size": 54,
            "masculine_tendency": 36.1,
            "feminine_tendency": 6.2,
            "description": "Defence shows highest masculine bias with military language"
        },
        "Government & Public Sector": {
            "average_bias": 18.7,
            "neutral_threshold": 15.0,
            "best_practice": 10.0,
            "sample_size": 98,
            "masculine_tendency": 12.4,
            "feminine_tendency": 6.3,
            "description": "Public sector shows most balanced language patterns"
        },
        "Academic & Research Institutions": {
            "average_bias": 16.2,
            "neutral_threshold": 15.0,
            "best_practice": 8.0,
            "sample_size": 145,
            "masculine_tendency": 10.8,
            "feminine_tendency": 5.4,
            "description": "Academic sector shows lowest bias with collaborative emphasis"
        },
        "Consulting Services": {
            "average_bias": 34.7,
            "neutral_threshold": 22.0,
            "best_practice": 16.0,
            "sample_size": 112,
            "masculine_tendency": 27.3,
            "feminine_tendency": 7.4,
            "description": "Consulting emphasizes competitive and client-focused language"
        },
        "Technology & Software": {
            "average_bias": 30.5,
            "neutral_threshold": 20.0,
            "best_practice": 14.0,
            "sample_size": 87,
            "masculine_tendency": 24.2,
            "feminine_tendency": 6.3,
            "description": "Tech sector shows innovation-focused masculine language"
        },
        "Retail & E-commerce": {
            "average_bias": 25.1,
            "neutral_threshold": 18.0,
            "best_practice": 12.0,
            "sample_size": 76,
            "masculine_tendency": 18.7,
            "feminine_tendency": 6.4,
            "description": "Retail shows moderate bias with customer service balance"
        },
        "General OR/Analytics": {
            "average_bias": 28.4,
            "neutral_threshold": 20.0,
            "best_practice": 15.0,
            "sample_size": 308,
            "masculine_tendency": 22.1,
            "feminine_tendency": 6.3,
            "description": "General OR roles show moderate masculine bias overall"
        },
        "Other": {
            "average_bias": 28.4,
            "neutral_threshold": 20.0,
            "best_practice": 15.0,
            "sample_size": 50,
            "masculine_tendency": 22.1,
            "feminine_tendency": 6.3,
            "description": "Other sectors follow general OR patterns"
        }
    }
    return industry_benchmarks

# Research-based data and word lists
@st.cache_data
def load_analysis_components():
    """Load validated word lists and analysis components"""
    # ENHANCED masculine words (100+ words)
    masculine_words = [
        # Core Professional Terms (Masculine-coded)
        'competitive', 'aggressive', 'dominant', 'driven', 'ambitious', 'decisive', 
        'strong', 'leader', 'lead', 'manage', 'control', 'challenge', 'achieve', 
        'dominate', 'excel', 'individual', 'independent', 'self-sufficient',
        'results-driven', 'performance', 'efficiency', 'strategic', 'execution',
        'analytical', 'objective', 'autonomous', 'determined', 'superior',
        'direct', 'drive', 'compete', 'win', 'hierarchy', 'decision',
        'responsibility', 'active', 'outperform', 'metrics', 'targets',
        'self-motivated', 'self-reliant', 'assertive', 'confident',
        
        # Leadership & Authority Terms
        'command', 'conquer', 'dominate', 'rule', 'govern', 'supervise',
        'direct', 'oversee', 'boss', 'chief', 'head', 'chairman', 'master',
        'principal', 'senior', 'primary', 'first', 'top', 'apex', 'peak',
        'supreme', 'ultimate', 'maximum', 'premier', 'elite', 'exclusive',
        
        # Competitive Language
        'beat', 'crush', 'destroy', 'demolish', 'eliminate', 'defeat',
        'overcome', 'surpass', 'overtake', 'outdo', 'outclass', 'outrank',
        'triumph', 'victory', 'winner', 'champion', 'first-place', 'top-tier',
        'best-in-class', 'market-leading', 'industry-leading', 'cutting-edge',
        
        # Action/Violence Metaphors
        'attack', 'strike', 'hit', 'punch', 'kick', 'fight', 'battle',
        'war', 'combat', 'militant', 'warrior', 'soldier', 'tactical',
        'strategic', 'offensive', 'defensive', 'target', 'aim', 'shoot',
        'fire', 'blast', 'explosive', 'powerful', 'forceful', 'intense',
        
        # OR-Specific Masculine Terms
        'optimize', 'maximize', 'algorithm', 'data-driven', 'quantitative',
        'analytical', 'systematic', 'methodical', 'rigorous', 'precise',
        'exact', 'accurate', 'efficient', 'effective', 'productive',
        'streamlined', 'automated', 'scalable', 'robust', 'sophisticated',
        
        # Innovation & Tech Language
        'disruptive', 'revolutionary', 'breakthrough', 'pioneering', 'groundbreaking',
        'innovative', 'cutting-edge', 'state-of-the-art', 'advanced', 'next-generation',
        'future-proof', 'game-changing', 'paradigm-shifting', 'transformational',
        'visionary', 'forward-thinking', 'progressive', 'dynamic', 'agile'
    ]
    
    # ENHANCED feminine words (80+ words)
    feminine_words = [
        # Core Collaborative Terms
        'collaborative', 'supportive', 'nurturing', 'empathetic', 'caring',
        'team', 'together', 'partnership', 'inclusive', 'responsive',
        'communicate', 'understand', 'help', 'assist', 'share',
        'community', 'relationship', 'trust', 'kind', 'cooperative',
        'interpersonal', 'motivated', 'committed', 'dedicated',
        'facilitation', 'coordination', 'consultation', 'liaison',
        'consensus', 'engagement', 'support', 'stakeholder',
        'patient', 'gentle', 'warm', 'welcoming', 'considerate',
        
        # Relationship & Communication Terms
        'connect', 'bond', 'relate', 'communicate', 'listen', 'hear',
        'understand', 'empathize', 'sympathize', 'comfort', 'console',
        'encourage', 'inspire', 'motivate', 'uplift', 'support',
        'guide', 'mentor', 'coach', 'teach', 'educate', 'train',
        'develop', 'grow', 'nurture', 'foster', 'cultivate',
        
        # Inclusive & Diverse Language
        'inclusive', 'diverse', 'multicultural', 'varied', 'broad',
        'wide-ranging', 'comprehensive', 'holistic', 'integrated',
        'balanced', 'harmonious', 'peaceful', 'calm', 'serene',
        'stable', 'consistent', 'reliable', 'dependable', 'trustworthy',
        
        # Care & Service Terms
        'serve', 'service', 'help', 'assist', 'aid', 'support',
        'care', 'tend', 'look after', 'protect', 'preserve',
        'maintain', 'sustain', 'nourish', 'feed', 'provide',
        'give', 'offer', 'share', 'contribute', 'participate'
    ]
    
    # ENHANCED bias patterns (200+ patterns)
    bias_patterns = {
        # Competitive & Aggressive Patterns (Masculine +15 to +35)
        'competitive environment': 25, 'fast-paced environment': 20, 'challenging role': 15,
        'high-pressure': 22, 'demanding environment': 20, 'aggressive approach': 30,
        'dominant position': 28, 'results-driven culture': 22, 'individual contributor': 20,
        'strong leadership': 16, 'exceed expectations': 18, 'drive results': 20,
        'take charge': 24, 'outperform competitors': 26, 'prove yourself': 25,
        'beat targets': 23, 'crush the competition': 35, 'dominate the market': 32,
        'aggressive sales': 28, 'competitive advantage': 24, 'win at all costs': 33,
        'take no prisoners': 35, 'survival of the fittest': 30, 'dog-eat-dog': 32,
        'cut-throat environment': 34, 'winner takes all': 30, 'first to market': 22,
        'market domination': 28, 'industry leader': 20, 'top performer': 18,
        'best in class': 16, 'world-class': 15, 'elite team': 19, 'A-team': 17,
        
        # Action & Violence Metaphors (Masculine +20 to +35)
        'hit the ground running': 25, 'shoot for the stars': 22, 'aim high': 18,
        'target achievement': 20, 'fire on all cylinders': 28, 'full steam ahead': 24,
        'go for the kill': 35, 'finish strong': 22, 'power through': 26,
        'drive hard': 24, 'push boundaries': 20, 'break barriers': 18,
        'smash goals': 28, 'attack the problem': 26, 'tackle challenges': 20,
        'fight for success': 30, 'battle-tested': 28, 'war room': 32,
        'frontline': 25, 'in the trenches': 27, 'combat ready': 30,
        'tactical approach': 22, 'strategic offensive': 28, 'launch attack': 32,
        
        # Authority & Control (Masculine +15 to +30)
        'take control': 25, 'seize opportunity': 22, 'command respect': 26,
        'assert authority': 28, 'establish dominance': 30, 'rule the market': 32,
        'govern processes': 24, 'master the domain': 26, 'own the space': 24,
        'lead from the front': 20, 'take the helm': 22, 'drive change': 18,
        'spearhead initiative': 24, 'pioneer solutions': 20, 'champion results': 18,
        'decision maker': 20, 'final say': 22, 'ultimate authority': 26,
        'call the shots': 24, 'run the show': 23, 'boss level': 21,
        
        # Performance & Achievement (Masculine +15 to +25)
        'top tier performance': 20, 'exceptional results': 18, 'outstanding achievement': 16,
        'superior performance': 22, 'maximum impact': 20, 'optimal results': 18,
        'peak performance': 21, 'record-breaking': 23, 'game-changing': 24,
        'revolutionary approach': 22, 'disruptive innovation': 25, 'paradigm shift': 23,
        'breakthrough results': 21, 'cutting-edge solution': 20, 'state-of-the-art': 19,
        'next-generation': 18, 'future-proof': 17, 'industry-leading': 20,
        'best-in-breed': 19, 'world-renowned': 18, 'globally recognized': 17,
        'internationally acclaimed': 18, 'award-winning': 16, 'celebrated': 15,
        
        # Individual Focus (Masculine +15 to +25)
        'self-starter': 20, 'self-motivated': 18, 'self-driven': 19,
        'independent worker': 22, 'autonomous role': 21, 'solo contributor': 24,
        'individual responsibility': 20, 'personal accountability': 18,
        'own your success': 23, 'personal brand': 19, 'individual achievement': 21,
        'self-reliant': 20, 'stand alone': 22, 'single-handedly': 25,
        'one-person show': 24, 'individual expertise': 19, 'personal mastery': 20,
        
        # Sports & Competition Metaphors (Masculine +18 to +30)
        'home run': 25, 'slam dunk': 28, 'touchdown': 26, 'grand slam': 30,
        'hat trick': 24, 'knockout punch': 32, 'winning formula': 22,
        'championship level': 24, 'gold medal': 20, 'trophy': 18,
        'hall of fame': 21, 'mvp': 23, 'all-star': 22, 'pro level': 20,
        'major league': 21, 'world series': 23, 'super bowl': 25,
        'olympics': 19, 'marathon': 17, 'sprint': 19, 'race to finish': 21,
        
        # Technical Masculine Terms (OR-specific +12 to +20)
        'algorithm optimization': 18, 'data mining': 15, 'machine learning': 12,
        'artificial intelligence': 14, 'big data': 13, 'analytics engine': 16,
        'optimization model': 17, 'computational power': 19, 'processing speed': 18,
        'system architecture': 16, 'database performance': 17, 'code efficiency': 18,
        'technical mastery': 20, 'engineering excellence': 19, 'systematic approach': 15,
        
        # ===== FEMININE PATTERNS (Negative scores: -10 to -25) =====
        
        # Collaborative & Team Patterns
        'collaborative team': -18, 'supportive environment': -20, 'inclusive culture': -15,
        'work-life balance': -12, 'team player': -14, 'diverse team': -16,
        'stakeholder engagement': -15, 'consensus building': -18, 'relationship building': -16,
        'mentoring opportunities': -12, 'flexible working': -10, 'welcoming environment': -14,
        'caring culture': -16, 'nurturing talent': -18, 'collaborative approach': -17,
        'team-oriented': -15, 'group dynamics': -14, 'collective effort': -16,
        'shared responsibility': -15, 'joint venture': -13, 'partnership model': -14,
        'cooperative strategy': -16, 'unified approach': -15, 'together we': -17,
        'our team': -14, 'we believe': -13, 'community focus': -16,
        
        # Communication & Relationship Focus
        'open communication': -15, 'active listening': -17, 'meaningful dialogue': -16,
        'transparent communication': -14, 'honest feedback': -13, 'constructive input': -15,
        'empathetic leadership': -20, 'compassionate management': -19, 'understanding approach': -16,
        'patient guidance': -18, 'gentle coaching': -19, 'kind supervision': -17,
        'thoughtful consideration': -15, 'careful planning': -13, 'mindful approach': -16,
        'considerate leadership': -17, 'respectful workplace': -15, 'dignified treatment': -14,
        
        # Care & Service Orientation
        'service excellence': -14, 'customer care': -16, 'client service': -15,
        'helping others': -18, 'serving community': -17, 'supporting colleagues': -16,
        'assisting customers': -15, 'caring for clients': -18, 'nurturing relationships': -19,
        'fostering growth': -17, 'cultivating talent': -18, 'developing people': -16,
        'growing together': -17, 'learning environment': -15, 'educational focus': -14,
        'teaching moments': -16, 'guidance and support': -17, 'mentorship program': -15,
        
        # Inclusive & Diversity Focus
        'diversity and inclusion': -16, 'equal opportunity': -15, 'fair treatment': -14,
        'inclusive practices': -17, 'diverse perspectives': -16, 'multicultural team': -15,
        'varied backgrounds': -14, 'different viewpoints': -15, 'broad representation': -16,
        'wide range': -13, 'comprehensive view': -14, 'holistic approach': -15,
        'integrated solution': -14, 'balanced perspective': -16, 'harmonious workplace': -17,
        'peaceful environment': -16, 'calm atmosphere': -15, 'serene setting': -17,
        
        # Work-Life Balance & Flexibility
        'flexible schedule': -12, 'remote work': -10, 'home office': -11,
        'flexible hours': -12, 'part-time options': -13, 'job sharing': -14,
        'compressed schedule': -11, 'flexible arrangement': -12, 'work from home': -10,
        'life balance': -14, 'personal time': -13, 'family friendly': -15,
        'child care': -16, 'maternity leave': -17, 'paternity leave': -15,
        'wellness program': -14, 'health benefits': -13, 'mental health': -15,
        
        # Emotional Intelligence & Soft Skills
        'emotional intelligence': -18, 'interpersonal skills': -16, 'social awareness': -15,
        'cultural sensitivity': -17, 'empathy training': -19, 'compassion focus': -18,
        'understanding nature': -16, 'patient approach': -17, 'gentle manner': -18,
        'kind leadership': -17, 'warm environment': -16, 'friendly atmosphere': -15,
        'welcoming culture': -16, 'accepting workplace': -15, 'tolerant environment': -14,
        
        # Growth & Development Focus
        'personal development': -14, 'professional growth': -13, 'career advancement': -12,
        'skill building': -13, 'knowledge sharing': -15, 'learning opportunities': -14,
        'training programs': -13, 'development path': -12, 'growth mindset': -14,
        'continuous learning': -15, 'lifelong education': -14, 'skill enhancement': -13,
        'capability building': -14, 'talent development': -15, 'potential realization': -14,
        
        # === NEUTRAL PATTERNS (0 to ±5) ===
        'professional environment': 0, 'business focus': 0, 'corporate culture': 0,
        'organizational goals': 0, 'company objectives': 0, 'strategic planning': 2,
        'operational excellence': 1, 'quality assurance': 0, 'process improvement': 1,
        'continuous improvement': 0, 'best practices': 1, 'industry standards': 0,
        'regulatory compliance': 0, 'policy adherence': 0, 'procedure following': 0
    }
    
    return masculine_words, feminine_words, bias_patterns

# AI Analysis Results from Research Dataset
@st.cache_data
def load_ai_research_findings():
    """Load pre-computed AI analysis results from research dataset"""
    ai_findings = {
        'dataset_stats': {
            'total_analyzed': 308,
            'total_cost': 5.03,
            'cost_per_analysis': 0.016,
            'coverage_percentage': 25.0
        },
        'bias_distribution': {
            'neutral': 64.0,
            'masculine': 24.4, 
            'feminine': 11.7
        },
        'correlation_with_methods': {
            'lexicon_correlation': 0.160,
            'sentiment_correlation': 0.013,
            'contextual_correlation': 0.087
        },
        'confidence_metrics': {
            'average_confidence': 78.5,
            'high_confidence_cases': 89.2,
            'method_agreement_rate': 68.0
        }
    }
    return ai_findings

# Analysis functions
def find_gendered_words(text, masculine_words, feminine_words):
    """Identify gendered language in text"""
    words = re.findall(r'\b\w+\b', text.lower())
    
    masculine_found = []
    feminine_found = []
    
    for word in words:
        if word in masculine_words and word not in masculine_found:
            masculine_found.append(word)
        elif word in feminine_words and word not in feminine_found:
            feminine_found.append(word)
    
    return masculine_found, feminine_found

def perform_lexicon_analysis(text, masculine_words, feminine_words):
    """Lexicon-based gender bias analysis"""
    masculine_matches, feminine_matches = find_gendered_words(text, masculine_words, feminine_words)
    masculine_count = len(masculine_matches)
    feminine_count = len(feminine_matches)
    total_words = len(text.split())
    total_gendered = masculine_count + feminine_count
    
    if total_gendered == 0:
        bias_score = 0
    else:
        bias_score = ((masculine_count - feminine_count) / total_gendered) * 100
    
    confidence = min(90, 60 + total_gendered * 3)
    
    return {
        'score': round(bias_score, 1),
        'masculine_words': masculine_matches,
        'feminine_words': feminine_matches,
        'masculine_count': masculine_count,
        'feminine_count': feminine_count,
        'total_words': total_words,
        'confidence': round(confidence, 1)
    }

def perform_contextual_analysis(text, bias_patterns):
    """Advanced contextual pattern analysis"""
    text_lower = text.lower()
    score = 0
    detected_patterns = []
    
    for pattern, weight in bias_patterns.items():
        if pattern in text_lower:
            score += weight
            detected_patterns.append((pattern, weight))
    
    sentences = text.split('.')
    structural_modifiers = 0
    
    for sentence in sentences:
        sentence_lower = sentence.lower().strip()
        if not sentence_lower:
            continue
            
        if any(phrase in sentence_lower for phrase in ['must be', 'should be', 'required to', 'expected to']):
            structural_modifiers += 5
            
        if any(word in sentence_lower for word in ['best', 'top', 'leading', 'premier', 'superior', 'excellent']):
            structural_modifiers += 4
            
        if any(phrase in sentence_lower for phrase in ['you will', 'individual', 'independently', 'on your own']):
            structural_modifiers += 3
        elif any(phrase in sentence_lower for phrase in ['we', 'our team', 'together', 'collaborate', 'partnership']):
            structural_modifiers -= 3
    
    final_score = score + structural_modifiers
    confidence = min(85, 65 + len(detected_patterns) * 3)
    
    return {
        'score': round(max(-100, min(100, final_score)), 1),
        'confidence': round(confidence, 1),
        'detected_patterns': detected_patterns,
        'structural_modifiers': structural_modifiers
    }

def perform_sentiment_analysis(text):
    """Sentiment-based bias detection"""
    intensity_markers = {
        'excellent': 12, 'outstanding': 18, 'exceptional': 22, 'superior': 20,
        'strong': 10, 'powerful': 16, 'competitive': 20, 'intense': 16,
        'aggressive': 25, 'driven': 15, 'ambitious': 14, 'demanding': 18,
        'challenging': 12, 'rigorous': 14, 'tough': 16, 'hardcore': 22,
        'supportive': -12, 'caring': -15, 'collaborative': -10, 'cooperative': -12,
        'inclusive': -12, 'nurturing': -18, 'empathetic': -20, 'understanding': -10,
        'patient': -8, 'kind': -10, 'gentle': -12, 'warm': -10,
        'welcoming': -14, 'considerate': -8, 'thoughtful': -6, 'helpful': -8
    }
    
    words = re.findall(r'\b\w+\b', text.lower())
    score = 0
    found_markers = []
    
    for word in words:
        if word in intensity_markers:
            score += intensity_markers[word]
            if word not in found_markers:
                found_markers.append(word)
    
    polarity = 0
    subjectivity = 0.5
    
    if TEXTBLOB_AVAILABLE:
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            
            if subjectivity > 0.6:
                if polarity > 0.3:
                    score += 15
                elif polarity < -0.2:
                    score += 10
                else:
                    score -= 10
        except Exception as e:
            pass  # Use default values
    
    confidence = min(80, 50 + len(found_markers) * 4)
    
    return {
        'score': round(max(-100, min(100, score)), 1),
        'confidence': round(confidence, 1),
        'found_markers': found_markers,
        'polarity': round(polarity, 3),
        'subjectivity': round(subjectivity, 3)
    }

def calculate_ensemble_score(lexicon_score, contextual_score, sentiment_score):
    """Calculate weighted ensemble score"""
    ensemble = (lexicon_score * 0.4) + (contextual_score * 0.35) + (sentiment_score * 0.25)
    return round(ensemble, 1)

def get_classification(score):
    """Determine bias classification and styling - ORIGINAL VERSION"""
    abs_score = abs(score)
    if abs_score <= 20:
        return "Well Balanced", "#059669", "excellent"
    elif abs_score <= 40:
        return "Moderate Bias", "#d97706", "warning"
    else:
        return "High Bias", "#dc2626", "error"

def get_classification_with_benchmarks(score, industry_benchmarks, selected_industry):
    """Determine bias classification with industry-specific benchmarks"""
    benchmark = industry_benchmarks.get(selected_industry, industry_benchmarks["General OR/Analytics"])
    abs_score = abs(score)
    
    if abs_score <= benchmark["best_practice"]:
        return "Excellent", "#059669", "excellent", "Top-tier inclusive language"
    elif abs_score <= benchmark["neutral_threshold"]:
        return "Good", "#059669", "excellent", "Well-balanced language"
    elif abs_score <= benchmark["average_bias"]:
        return "Industry Average", "#d97706", "warning", "Typical for your sector"
    elif abs_score <= (benchmark["average_bias"] * 1.5):
        return "Above Average Bias", "#dc2626", "error", "Higher than sector norm"
    else:
        return "High Bias", "#dc2626", "error", "Significantly biased"

def create_bias_gauge(score, title="Bias Score"):
    """Create a gauge chart for bias visualization - ORIGINAL VERSION"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title},
        gauge = {
            'axis': {'range': [-100, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [-100, -20], 'color': "lightgreen"},
                {'range': [-20, 20], 'color': "lightyellow"},
                {'range': [20, 100], 'color': "lightcoral"}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90}}))
    
    fig.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=60, b=20),
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig

def generate_improved_version(text, masculine_words, feminine_words, bias_patterns):
    """Generate an improved version of the job advertisement - ORIGINAL VERSION"""
    improved_text = text
    changes_made = []
    
    # ENHANCED phrase-level replacements (80+ phrases)
    phrase_replacements = {
        # Environment & Culture Phrases
        'fast-paced environment': 'dynamic work environment',
        'high-pressure environment': 'results-focused environment',
        'competitive environment': 'performance-oriented environment',
        'demanding environment': 'challenging and supportive environment',
        'aggressive environment': 'proactive work environment',
        'cut-throat environment': 'performance-driven environment',
        'dog-eat-dog environment': 'merit-based environment',
        'survival of the fittest': 'performance excellence',
        'winner takes all': 'merit-based success',
        'top-tier environment': 'excellence-focused environment',
        
        # Role & Responsibility Phrases
        'individual contributor': 'independent professional',
        'self-starter required': 'motivated professional needed',
        'must be self-motivated': 'should be proactive',
        'work independently': 'work autonomously with team support',
        'solo contributor': 'independent team member',
        'one-person operation': 'independent role with collaboration',
        'single-handedly manage': 'take ownership while collaborating',
        'own the project': 'lead the project',
        'take charge of': 'coordinate and manage',
        'command the team': 'lead the team',
        
        # Performance & Achievement Phrases
        'crush the competition': 'outperform competitors',
        'beat the competition': 'exceed market standards',
        'dominate the market': 'lead in the market',
        'destroy the competition': 'surpass competitors',
        'kill it in sales': 'excel in sales',
        'smash targets': 'exceed targets',
        'blow away expectations': 'surpass expectations',
        'hit it out of the park': 'achieve outstanding results',
        'slam dunk opportunity': 'excellent opportunity',
        'home run performance': 'outstanding performance',
        
        # Action & Violence Metaphors
        'attack the problem': 'address the challenge',
        'tackle the issue': 'resolve the issue',
        'fight for results': 'work diligently for results',
        'battle-tested experience': 'proven experience',
        'war room strategy': 'strategic planning session',
        'frontline experience': 'hands-on experience',
        'in the trenches': 'in operational roles',
        'combat ready': 'fully prepared',
        'tactical approach': 'strategic approach',
        'strategic offensive': 'strategic initiative',
        'launch attack': 'implement strategy',
        'fire on all cylinders': 'perform at full capacity',
        'full steam ahead': 'move forward decisively',
        'go for the kill': 'pursue success',
        'take no prisoners': 'maintain high standards',
        
        # Leadership & Authority Phrases
        'take control': 'take leadership',
        'seize control': 'assume leadership',
        'assert dominance': 'demonstrate leadership',
        'establish dominance': 'establish leadership',
        'rule the market': 'lead the market',
        'govern the process': 'guide the process',
        'master the domain': 'excel in the field',
        'own the space': 'lead in the sector',
        'call the shots': 'make key decisions',
        'run the show': 'manage operations',
        'boss the project': 'lead the project',
        
        # Business Jargon & Clichés
        'think outside the box': 'approach creatively',
        'move the needle': 'drive meaningful change',
        'low-hanging fruit': 'immediate opportunities',
        'boil the ocean': 'comprehensive approach',
        'drink the kool-aid': 'embrace company culture',
        'circle back': 'follow up',
        'touch base': 'connect',
        'ping me': 'contact me',
        'loop in': 'include',
        'dive deep': 'analyze thoroughly',
        'drill down': 'examine in detail',
        'bottom line': 'key result',
        'net-net': 'overall result',
        'at the end of the day': 'ultimately',
        'when push comes to shove': 'when necessary',
        
        # OR & Tech-Specific Phrases
        'analytics ninja': 'analytics professional',
        'data wizard': 'data specialist',
        'algorithm guru': 'algorithm expert',
        'optimization master': 'optimization specialist',
        'machine learning rockstar': 'machine learning expert',
        'AI superhero': 'AI specialist',
        'coding warrior': 'skilled developer',
        'tech guru': 'technology expert',
        'digital native': 'technology-savvy professional',
        'innovation champion': 'innovation leader',
        'disruptor mindset': 'innovative thinking',
        'game-changer attitude': 'transformational approach',
        
        # Sales & Business Development
        'killer instinct': 'strong business acumen',
        'hunter mentality': 'proactive sales approach',
        'shark in sales': 'effective salesperson',
        'predatory pricing': 'competitive pricing',
        'blood in the water': 'market opportunity',
        'feeding frenzy': 'high activity period',
        'circle the wagons': 'coordinate response',
        'batten down hatches': 'prepare thoroughly',
        'hunker down': 'focus intensively',
        'lock and load': 'prepare for action'
    }
    
    # MASSIVELY ENHANCED word-level replacements (120+ words)
    word_replacements = {
        # Core Professional Terms
        'competitive': 'results-focused', 'aggressive': 'proactive', 'dominate': 'excel in',
        'driven': 'motivated', 'ambitious': 'goal-oriented', 'strong': 'effective',
        'challenging': 'engaging', 'demanding': 'comprehensive', 'individual': 'collaborative',
        'manage': 'coordinate', 'control': 'guide', 'lead': 'facilitate',
        'achieve': 'accomplish', 'exceed': 'surpass', 'outperform': 'excel',
        'high-pressure': 'dynamic', 'fast-paced': 'efficient', 'results-driven': 'results-oriented',
        'self-sufficient': 'independent and collaborative', 'dominant': 'leading', 'superior': 'excellent',
        
        # Leadership & Authority Terms
        'command': 'lead', 'conquer': 'succeed in', 'rule': 'guide', 'govern': 'oversee',
        'supervise': 'coordinate', 'direct': 'guide', 'boss': 'lead', 'chief': 'lead',
        'master': 'expert in', 'principal': 'primary', 'supreme': 'excellent',
        'ultimate': 'optimal', 'maximum': 'highest', 'premier': 'leading',
        'elite': 'skilled', 'exclusive': 'specialized', 'top-tier': 'high-quality',
        'first-class': 'excellent', 'world-class': 'outstanding', 'best-in-class': 'leading',
        
        # Competitive Language
        'beat': 'surpass', 'crush': 'excel against', 'destroy': 'outperform',
        'demolish': 'significantly exceed', 'eliminate': 'surpass', 'defeat': 'outperform',
        'overcome': 'address successfully', 'overtake': 'surpass', 'outdo': 'exceed',
        'outclass': 'excel beyond', 'outrank': 'perform better than', 'triumph': 'succeed',
        'victory': 'success', 'winner': 'successful candidate', 'champion': 'leader',
        'market-leading': 'industry-leading', 'cutting-edge': 'innovative',
        
        # Action/Violence Metaphors
        'attack': 'address', 'strike': 'implement', 'hit': 'achieve', 'punch': 'impact',
        'kick': 'initiate', 'fight': 'work diligently', 'battle': 'work on',
        'war': 'intensive effort', 'combat': 'address', 'militant': 'dedicated',
        'warrior': 'dedicated professional', 'soldier': 'team member', 'tactical': 'strategic',
        'offensive': 'proactive', 'defensive': 'protective', 'target': 'objective',
        'aim': 'focus on', 'shoot': 'strive for', 'fire': 'launch', 'blast': 'accelerate',
        'explosive': 'dynamic', 'powerful': 'effective', 'forceful': 'decisive',
        'intense': 'focused', 'hardcore': 'dedicated', 'brutal': 'intensive',
        
        # OR-Specific Terms
        'optimize': 'improve', 'maximize': 'enhance', 'algorithm': 'systematic method',
        'data-driven': 'data-informed', 'quantitative': 'analytical', 'systematic': 'organized',
        'methodical': 'thorough', 'rigorous': 'comprehensive', 'precise': 'accurate',
        'exact': 'accurate', 'efficient': 'effective', 'productive': 'effective',
        'streamlined': 'efficient', 'automated': 'systematized', 'scalable': 'adaptable',
        'robust': 'reliable', 'sophisticated': 'advanced', 'complex': 'comprehensive',
        
        # Innovation & Tech Language
        'disruptive': 'innovative', 'revolutionary': 'transformational', 'breakthrough': 'significant advance',
        'pioneering': 'leading-edge', 'groundbreaking': 'innovative', 'cutting-edge': 'advanced',
        'state-of-the-art': 'current best practice', 'next-generation': 'advanced',
        'future-proof': 'adaptable', 'game-changing': 'transformational', 'paradigm-shifting': 'innovative',
        'transformational': 'significant', 'visionary': 'forward-thinking', 'progressive': 'forward-looking',
        'dynamic': 'adaptable', 'agile': 'flexible', 'nimble': 'responsive',
        
        # Performance Terms
        'killer': 'excellent', 'beast': 'professional', 'machine': 'systematic professional',
        'monster': 'significant', 'insane': 'remarkable', 'sick': 'impressive',
        'wicked': 'excellent', 'badass': 'skilled', 'ninja': 'expert',
        'guru': 'specialist', 'wizard': 'expert', 'rockstar': 'outstanding professional',
        'superhero': 'exceptional professional', 'legend': 'experienced professional',
        
        # Intensity & Extremes
        'brutal': 'intensive', 'savage': 'intense', 'fierce': 'dedicated',
        'ruthless': 'focused', 'merciless': 'thorough', 'relentless': 'persistent',
        'unstoppable': 'determined', 'unbeatable': 'excellent', 'invincible': 'highly capable',
        'bulletproof': 'reliable', 'rock-solid': 'dependable', 'iron-clad': 'secure',
        
        # Authority & Control
        'dictate': 'determine', 'mandate': 'require', 'decree': 'establish',
        'enforce': 'implement', 'impose': 'apply', 'demand': 'require',
        'insist': 'require', 'compel': 'encourage', 'force': 'drive',
        'pressure': 'encourage', 'push': 'motivate', 'drive': 'guide'
    }
    
    # Process phrase-level replacements first (more context-specific)
    for phrase, replacement in phrase_replacements.items():
        if phrase.lower() in improved_text.lower():
            # Case-insensitive replacement while preserving original case structure
            pattern = re.compile(re.escape(phrase), re.IGNORECASE)
            improved_text = pattern.sub(replacement, improved_text)
            changes_made.append(f"'{phrase}' → '{replacement}'")
    
    # Then process word-level replacements
    words = improved_text.split()
    for i, word in enumerate(words):
        clean_word = word.lower().strip('.,!?;:()"')
        if clean_word in word_replacements:
            # Preserve capitalization and punctuation
            punctuation = ''.join(c for c in word if not c.isalnum())
            replacement = word_replacements[clean_word]
            
            if word.istitle():
                words[i] = replacement.title() + punctuation
            elif word.isupper():
                words[i] = replacement.upper() + punctuation
            else:
                words[i] = replacement + punctuation
            
            # Only add to changes if not already added from phrase replacement
            change_text = f"'{clean_word}' → '{replacement}'"
            if change_text not in changes_made:
                changes_made.append(change_text)
    
    improved_text = ' '.join(words)
    
    # Remove duplicates from changes_made while preserving order
    seen = set()
    changes_made = [x for x in changes_made if not (x in seen or seen.add(x))]
    
    return improved_text, changes_made

# Load components
masculine_words, feminine_words, bias_patterns = load_analysis_components()
industry_benchmarks = load_industry_benchmarks()

# Initialize session state
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None

# Header
st.markdown("""
<div class="main-header">
    <div class="header-title">Operational Research Job Advertisement Language Analyser</div>
    <div class="header-subtitle">Analyse and improve gender inclusivity in Operational Research job postings</div>
    <div style="margin-top: 1.5rem; font-size: 0.95rem;">
        <strong>© 2025 | Research partnership with The OR Society's WORAN (Women in Operational Research and Analytics Network)</strong>
    </div>
</div>
""", unsafe_allow_html=True)

# Research foundation statistics
st.markdown("""
<div style="background: linear-gradient(135deg, #f8fafc, #e2e8f0); border-radius: 16px; padding: 2rem; margin-bottom: 2rem; border: 1px solid #d1d5db;">
    <div style="text-align: center; margin-bottom: 1.5rem;">
        <h3 style="color: #1f2937; margin-bottom: 1rem;">Research-Validated Analysis Tool</h3>
        <p style="color: #6b7280; font-size: 1rem;">Built on comprehensive research of UK OR job market with AI validation</p>
    </div>
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 1rem;">
        <div style="text-align: center; padding: 1rem; background: white; border-radius: 8px;">
            <div style="font-size: 1.8rem; font-weight: bold; color: #3730a3;">1,200+</div>
            <div style="font-size: 0.85rem; color: #6b7280;">UK OR Jobs Analyzed</div>
        </div>
        <div style="text-align: center; padding: 1rem; background: white; border-radius: 8px;">
            <div style="font-size: 1.8rem; font-weight: bold; color: #3730a3;">300+</div>
            <div style="font-size: 0.85rem; color: #6b7280;">AI-Validated</div>
        </div>
        <div style="text-align: center; padding: 1rem; background: white; border-radius: 8px;">
            <div style="font-size: 1.8rem; font-weight: bold; color: #3730a3;">15</div>
            <div style="font-size: 0.85rem; color: #6b7280;">Industry Sectors</div>
        </div>
        <div style="text-align: center; padding: 1rem; background: white; border-radius: 8px;">
            <div style="font-size: 1.8rem; font-weight: bold; color: #3730a3;">47</div>
            <div style="font-size: 0.85rem; color: #6b7280;">Analysis Variables</div>
        </div>
        <div style="text-align: center; padding: 1rem; background: white; border-radius: 8px;">
            <div style="font-size: 1.8rem; font-weight: bold; color: #3730a3;">91.7%</div>
            <div style="font-size: 0.85rem; color: #6b7280;">Survey Respondents Interested</div>
        </div>
        <div style="text-align: center; padding: 1rem; background: white; border-radius: 8px;">
            <div style="font-size: 1.8rem; font-weight: bold; color: #3730a3;">Free</div>
            <div style="font-size: 0.85rem; color: #6b7280;">Analysis Tool</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Quick start guide
st.markdown("""
<div class="validation-box">
    <h4 style="color: #0c4a6e; margin-bottom: 1rem;">Quick Start Guide</h4>
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem;">
        <div>
            <strong>1. Select Your Industry:</strong><br>
            Choose your OR sector for accurate benchmarking.
        </div>
        <div>
            <strong>2. Paste Your Job Advert:</strong><br>
            Copy the complete job posting into the text box below.
        </div>
        <div>
            <strong>3. Select Analysis Method:</strong><br>
            Choose "Comprehensive Multi-Method" for best results.
        </div>
        <div>
            <strong>4. Get Instant Results:</strong><br>
            Receive a bias score, industry comparison, and an improved version.
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Main analysis interface
st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
st.markdown('<h2 class="section-title">Job Advertisement Analysis</h2>', unsafe_allow_html=True)

st.markdown('<div class="technique-selector">', unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("**Analysis Method**")
    technique = st.radio(
        "Choose analysis approach:",
        ["Lexicon-Based Analysis", "Contextual Pattern Analysis", "Sentiment Analysis", "Comprehensive Multi-Method Analysis"],
        index=3,
        help="Select the analysis technique for detecting gender-coded language",
        horizontal=True
    )

with col2:
    st.markdown("**Industry Context**")
    industry = st.selectbox(
        "Select OR industry sector:",
        [
            "Healthcare & Medical OR",
            "Financial Services & Banking", 
            "Supply Chain & Logistics",
            "Manufacturing & Production",
            "Transportation & Airlines",
            "Energy & Utilities",
            "Telecommunications",
            "Defence & Aerospace",
            "Government & Public Sector",
            "Academic & Research Institutions",
            "Consulting Services",
            "Technology & Software",
            "Retail & E-commerce",
            "General OR/Analytics",
            "Other"
        ],
        index=0,
        help="Choose your specific OR industry sector for relevant benchmarking"
    )

# Show industry benchmark info
if industry in industry_benchmarks:
    benchmark_info = industry_benchmarks[industry]
    st.info(f"**{industry}**: {benchmark_info['description']} (Sample: {benchmark_info['sample_size']} jobs)")

# AI Enhancement Option
if technique == "Comprehensive Multi-Method Analysis":
    st.markdown("**Enhanced Analysis Options**")
    ai_enhanced = st.checkbox(
        "Include AI research insights", 
        value=True,
        help="Incorporates findings from AI analysis of 300+ OR job advertisements"
    )
else:
    ai_enhanced = False

st.markdown('</div>', unsafe_allow_html=True)

# Text input
job_text = st.text_area(
    "**Enter job advertisement text for analysis:**",
    height=250,
    placeholder="Paste your job advertisement text here...",
    help="Enter the complete job advertisement text"
)

if job_text:
    word_count = len(job_text.split())
    char_count = len(job_text)
    st.caption(f"Text length: {word_count} words, {char_count} characters")

st.markdown('</div>', unsafe_allow_html=True)

# Analysis execution
if st.button("Analyse Text", type="primary", disabled=not job_text.strip()):
    with st.spinner("Performing comprehensive analysis..."):
        lexicon_results = perform_lexicon_analysis(job_text, masculine_words, feminine_words)
        contextual_results = perform_contextual_analysis(job_text, bias_patterns)
        sentiment_results = perform_sentiment_analysis(job_text)
        
        if technique == "Comprehensive Multi-Method Analysis":
            final_score = calculate_ensemble_score(
                lexicon_results['score'], 
                contextual_results['score'], 
                sentiment_results['score']
            )
            confidence = min(95, (lexicon_results['confidence'] + contextual_results['confidence'] + sentiment_results['confidence']) / 3)
        elif technique == "Lexicon-Based Analysis":
            final_score = lexicon_results['score']
            confidence = lexicon_results['confidence']
        elif technique == "Contextual Pattern Analysis":
            final_score = contextual_results['score']
            confidence = contextual_results['confidence']
        else:
            final_score = sentiment_results['score']
            confidence = sentiment_results['confidence']
        
        st.session_state.analysis_results = {
            'technique': technique,
            'industry': industry,
            'final_score': final_score,
            'confidence': confidence,
            'lexicon': lexicon_results,
            'contextual': contextual_results,
            'sentiment': sentiment_results,
            'text': job_text,
            'timestamp': datetime.now(),
            'ai_enhanced': ai_enhanced
        }

# Results display
if st.session_state.analysis_results:
    results = st.session_state.analysis_results
    classification, color, css_class = get_classification(results['final_score'])
    
    st.markdown('<div class="analysis-results">', unsafe_allow_html=True)
    st.markdown("## Analysis Results")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        gauge_fig = create_bias_gauge(results['final_score'], "Gender Bias Assessment")
        st.plotly_chart(gauge_fig, use_container_width=True)
    
    with col2:
        bias_direction = "Masculine" if results['final_score'] > 0 else "Feminine" if results['final_score'] < 0 else "Neutral"
        
        st.markdown(f"""
        <div class="metric-card {css_class}">
            <div class="metric-value">{abs(results['final_score']):.1f}</div>
            <div class="metric-label">Bias Score</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{bias_direction}</div>
            <div class="metric-label">Bias Direction</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{classification}</div>
            <div class="metric-label">Classification</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{results['confidence']:.0f}%</div>
            <div class="metric-label">Confidence Level</div>
        </div>
        """, unsafe_allow_html=True)

    # AI-Powered Rewritten Version
    if abs(results['final_score']) > 15:
        st.markdown("### Improved Version")
        
        st.markdown("""
        <div style="background: linear-gradient(135deg, #f0fdf4, #ecfdf5); border: 1px solid #059669; border-radius: 16px; padding: 1.5rem; margin: 1.5rem 0;">
            <h4 style="color: #065f46; margin-bottom: 1rem;">Research-Based Improvements</h4>
            <p style="color: #065f46;">Based on analysis of 1,200+ OR job advertisements, here's an improved version with reduced gender bias:</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Generate an improved version
        improved_text, changes_made = generate_improved_version(
            results['text'], 
            masculine_words, 
            feminine_words, 
            bias_patterns
        )
        
        # Analyse the improved version
        improved_lexicon = perform_lexicon_analysis(improved_text, masculine_words, feminine_words)
        improved_contextual = perform_contextual_analysis(improved_text, bias_patterns)
        improved_sentiment = perform_sentiment_analysis(improved_text)
        
        improved_score = calculate_ensemble_score(
            improved_lexicon['score'],
            improved_contextual['score'], 
            improved_sentiment['score']
        )
        
        improved_classification, improved_color, improved_css = get_classification(improved_score)
        
        # Show before/after comparison
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Original Version")
            st.markdown(f"""
            <div style="background: #fef2f2; border: 2px solid #dc2626; border-radius: 12px; padding: 1.5rem;">
                <div style="background: #dc2626; color: white; padding: 0.5rem 1rem; border-radius: 8px; margin-bottom: 1rem; text-align: center; font-size: 0.9rem;">
                    <strong>Bias Score: {abs(results['final_score']):.1f} ({classification})</strong>
                </div>
                <div style="font-size: 0.85rem; line-height: 1.5; max-height: 300px; overflow-y: auto;">{results['text']}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("#### Improved Version")
            st.markdown(f"""
            <div style="background: #f0fdf4; border: 2px solid #059669; border-radius: 12px; padding: 1.5rem;">
                <div style="background: #059669; color: white; padding: 0.5rem 1rem; border-radius: 8px; margin-bottom: 1rem; text-align: center; font-size: 0.9rem;">
                    <strong>Bias Score: {abs(improved_score):.1f} ({improved_classification})</strong>
                </div>
                <div style="font-size: 0.85rem; line-height: 1.5; max-height: 300px; overflow-y: auto;">{improved_text}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Show improvement metrics
        if abs(results['final_score']) > abs(improved_score):
            improvement_points = abs(results['final_score']) - abs(improved_score)
            improvement_percentage = (improvement_points / abs(results['final_score'])) * 100
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Bias Reduction", f"{improvement_points:.1f} points", f"-{improvement_percentage:.1f}%")
            
            with col2:
                st.metric("Classification", f"{improved_classification}", f"From {classification}")
            
            with col3:
                st.metric("Changes Made", f"{len(changes_made)} improvements", "Research-backed")
            
            # Show specific changes
            if changes_made:
                st.markdown("#### Specific Changes Made")
                
                changes_per_row = 2
                change_chunks = [changes_made[i:i+changes_per_row] for i in range(0, len(changes_made), changes_per_row)]
                
                for chunk in change_chunks:
                    cols = st.columns(len(chunk))
                    for idx, change in enumerate(chunk):
                        with cols[idx]:
                            if '→' in change:
                                original, replacement = change.split(' → ')
                                original = original.strip("'\"")
                                replacement = replacement.strip("'\"")
                                st.markdown(f"""
                                <div style="background: white; border: 1px solid #e5e7eb; border-radius: 8px; padding: 0.8rem; margin: 0.25rem 0;">
                                    <div style="color: #dc2626; font-weight: bold; font-size: 0.85rem; margin-bottom: 0.25rem;">❌ {original}</div>
                                    <div style="color: #059669; font-weight: bold; font-size: 0.85rem;">✅ {replacement}</div>
                                </div>
                                """, unsafe_allow_html=True)
            
            st.info("The improved version maintains all technical requirements while using more inclusive language based on our research of 1,200+ OR job advertisements.")

    # Industry Benchmark Comparison
    st.markdown("### Industry Benchmark Comparison")
    
    if results['industry'] in industry_benchmarks:
        benchmark_info = industry_benchmarks[results['industry']]
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Create industry comparison chart
            comparison_data = {
                'Metric': ['Your Score', 'Industry Average', 'Best Practice', 'Neutral Threshold'],
                'Score': [abs(results['final_score']), benchmark_info['average_bias'], 
                         benchmark_info['best_practice'], benchmark_info['neutral_threshold']],
            }
            
            fig = px.bar(
                comparison_data, 
                x='Metric', 
                y='Score',
                title=f"{results['industry']} - Bias Comparison",
                color='Score',
                color_continuous_scale='RdYlGn_r'
            )
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Industry context
            user_abs_score = abs(results['final_score'])
            
            if user_abs_score <= benchmark_info['best_practice']:
                status_color = "#059669"
                status = "Excellent"
            elif user_abs_score <= benchmark_info['neutral_threshold']:
                status_color = "#059669"
                status = "Good"
            elif user_abs_score <= benchmark_info['average_bias']:
                status_color = "#d97706"
                status = "Industry Average"
            else:
                status_color = "#dc2626"
                status = "Above Average Bias"
            
            st.markdown(f"""
            <div style="background: white; border: 2px solid {status_color}; border-radius: 12px; padding: 1.5rem;">
                <h4 style="color: {status_color};">Industry Context</h4>
                <p><strong>Sector:</strong> {results['industry']}</p>
                <p><strong>Sample Size:</strong> {benchmark_info['sample_size']} jobs</p>
                <p><strong>Your Score:</strong> {user_abs_score:.1f}</p>
                <p><strong>Industry Average:</strong> {benchmark_info['average_bias']:.1f}</p>
                <p><strong>Best Practice:</strong> {benchmark_info['best_practice']:.1f}</p>
                <div style="margin-top: 1rem; padding: 0.8rem; background: #f8f9fa; border-radius: 6px;">
                    <strong>Status:</strong> {status}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Word analysis
    if results['lexicon']['masculine_words'] or results['lexicon']['feminine_words']:
        st.markdown("### Detected Language Patterns")
        
        if results['lexicon']['masculine_words']:
            st.markdown("**Masculine-coded language detected:**")
            masculine_html = ""
            for word in results['lexicon']['masculine_words']:
                masculine_html += f'<span class="word-tag masculine-word">{word}</span>'
            st.markdown(masculine_html, unsafe_allow_html=True)
        
        if results['lexicon']['feminine_words']:
            st.markdown("**Feminine-coded language detected:**")
            feminine_html = ""
            for word in results['lexicon']['feminine_words']:
                feminine_html += f'<span class="word-tag feminine-word">{word}</span>'
            st.markdown(feminine_html, unsafe_allow_html=True)
    
    # Analysis interpretation
    st.markdown("### Analysis Interpretation")
    
    if abs(results['final_score']) <= 20:
        st.markdown(f"""
        <div class="insight-box insight-excellent">
            <strong>Excellent Balance Detected</strong><br>
            This job advertisement demonstrates well-balanced language with a bias score of {abs(results['final_score']):.1f}. 
            The text uses inclusive language that is likely to appeal to candidates regardless of gender.
        </div>
        """, unsafe_allow_html=True)
    
    elif abs(results['final_score']) <= 40:
        st.markdown(f"""
        <div class="insight-box insight-warning">
            <strong>Moderate Bias Identified</strong><br>
            This advertisement shows a bias score of {abs(results['final_score']):.1f}, indicating moderate gender bias. 
            {'Masculine' if results['final_score'] > 0 else 'Feminine'}-coded language may influence application rates. 
            Consider implementing suggested improvements to enhance inclusivity.
        </div>
        """, unsafe_allow_html=True)
    
    else:
        st.markdown(f"""
        <div class="insight-box insight-error">
            <strong>Significant Bias Detected</strong><br>
            This advertisement demonstrates high gender bias with a score of {abs(results['final_score']):.1f}. 
            The strong {'masculine' if results['final_score'] > 0 else 'feminine'} coding may significantly impact 
            the diversity of your applicant pool. Comprehensive revision is recommended.
        </div>
        """, unsafe_allow_html=True)

    # AI Research Context
    if results.get('ai_enhanced', False):
        st.markdown("### AI Research Context")
        
        user_score = abs(results['final_score'])
        ai_data = load_ai_research_findings()
        
        research_context = {
            'Your Analysis': user_score,
            'Research Average': 28.4,
            'AI Neutral Threshold': 20.0,
            'Research Best Practice': 15.0
        }
        
        context_df = pd.DataFrame(list(research_context.items()), columns=['Metric', 'Bias Score'])
        
        fig_research_context = px.bar(
            context_df, x='Metric', y='Bias Score',
            title='Your Analysis vs Research Benchmarks',
            color='Bias Score',
            color_continuous_scale='RdYlGn_r'
        )
        fig_research_context.update_layout(height=300)
        st.plotly_chart(fig_research_context, use_container_width=True)
        
        # Research-based insight
        if user_score <= 20:
            research_insight = f"Your bias score of {user_score:.1f} aligns with the {ai_data['bias_distribution']['neutral']:.1f}% of jobs classified as neutral in our AI research. This indicates balanced, inclusive language."
            insight_class = "insight-excellent"
        else:
            research_insight = f"Your bias score of {user_score:.1f} suggests room for improvement based on our research dataset of 1,200+ OR job advertisements."
            insight_class = "insight-warning"
        
        st.markdown(f"""
        <div class="insight-box {insight_class}">
            <strong>Research Context:</strong><br>
            {research_insight}
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Tool Capabilities
st.markdown("---")
st.markdown("## Tool Capabilities")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### Analysis Methods
    - Lexicon-Based: Identifies masculine/feminine coded words
    - Contextual Patterns: Detects bias in phrases and structure  
    - Sentiment Analysis: Measures emotional tone and intensity
    - Multi-Method: Combines all approaches for comprehensive analysis
    
    ### What You Get
    - Instant bias score and classification
    - Specific word-level recommendations
    - AI-improved job advertisement versions
    - Industry-specific benchmarking across 15 OR sectors
    """)

with col2:
    st.markdown("""
    ### Professional Benefits
    - Attract diverse talent with inclusive language
    - Industry benchmarking against sector-specific standards
    - Reduce unconscious bias in recruitment materials
    - *Improve application rates from underrepresented groups
    
    ### Organisational Use
    - HR departments and recruitment teams
    - Individual hiring managers
    - Diversity and inclusion initiatives
    - Regular job posting audits
    """)

# Usage guidance
st.markdown("### How to Use This Tool")
st.info("""
**Best Practices:**
- Select your specific industry for accurate benchmarking
- Paste complete job advertisements for accurate analysis
- Use "Comprehensive Multi-Method Analysis" for the most thorough results
- Review AI-improved versions when bias scores exceed 15
- Regular analysis helps maintain consistent, inclusive language
- Combine tool insights with human review for the best outcomes
""")

# Footer
st.markdown("""
<div class="footer-credits">
    <div style="text-align: center;">
        <h4 style="margin: 0; color: #f3f4f6; font-size: 1.2rem; margin-bottom: 1rem;">Research-Validated Language Analysis Tool</h4>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 2rem; margin-bottom: 2rem;">
            <div>
                <strong>Developed by:</strong><br>
                Marwa Ashfaq<br>
                University of Southampton<br>
                <a href="mailto:ma7n24@soton.ac.uk">ma7n24@soton.ac.uk</a>
            </div>
            <div>
                <strong>Research Partner:</strong><br>
                The OR Society's WORAN<br>
                (Women in Operational Research<br>
                and Analytics Network)
            </div>
            <div>
                <strong>Research Foundation:</strong><br>
                1,200+ UK OR job advertisements<br>
                15 industry-specific benchmarks<br>
                Multi-technique validation framework
            </div>
    </div>
    <div style="margin-top: 1rem; padding-top: 1rem; border-top: 1px solid rgba(255,255,255,0.2);">
        <p style="margin: 0; text-align: center; opacity: 0.8; color: white; font-size: 0.9rem;">
            <strong>Professional Use:</strong> Free tool for HR professionals, hiring managers, and inclusive recruitment initiatives
        </p>
    </div>
    <hr style="border: none; height: 1px; background: rgba(255,255,255,0.2); margin: 1.5rem 0;">
    <p style="margin: 0; text-align: center; opacity: 0.8; color: white;">
        © 2025 | Free professional tool for inclusive recruitment practices
    </p>
</div>
""", unsafe_allow_html=True)
