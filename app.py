import streamlit as st
import requests
import json
from datetime import datetime, timedelta
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from transformers import pipeline
import torch
from textblob import TextBlob
from snownlp import SnowNLP
from langdetect import detect, LangDetectException
import re
import os
from typing import List, Dict, Tuple

# Try to load .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, continue without it

# Page configuration
st.set_page_config(
    page_title="Hong Kong FIRE Stock News Summarizer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .news-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .sentiment-positive {
        color: #28a745;
        font-weight: bold;
    }
    .sentiment-negative {
        color: #dc3545;
        font-weight: bold;
    }
    .sentiment-neutral {
        color: #6c757d;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

class StockNewsSummarizer:
    def __init__(self):
        # Use the provided NewsAPI key directly
        self.newsapi_key = "9100f9fe1f4240b9a13e821c84f7c120"
        self.summarizer = None
        self._load_summarizer()
    
    def set_api_key(self, api_key: str):
        """Set the NewsAPI key"""
        self.newsapi_key = api_key
    
    def _load_summarizer(self):
        """Load the BART summarization model"""
        try:
            with st.spinner("Loading AI summarization model..."):
                self.summarizer = pipeline(
                    "summarization",
                    model="facebook/bart-large-cnn",
                    device=0 if torch.cuda.is_available() else -1  # Use CPU for Mac ARM
                )
            st.success("✅ AI model loaded successfully!")
        except Exception as e:
            st.error(f"❌ Failed to load AI model: {str(e)}")
            self.summarizer = None
    
    def get_news(self, symbol: str, days_back: int = 7) -> List[Dict]:
        """Fetch news from NewsAPI"""
        if not self.newsapi_key:
            st.error("❌ NewsAPI key not found. Please set NEWSAPI_KEY environment variable.")
            return []
        
        try:
            # Calculate date range
            to_date = datetime.now()
            from_date = to_date - timedelta(days=days_back)
            
            # Build query with typhoon filter for real estate stocks
            query = symbol
            if self._is_real_estate_stock(symbol):
                query += " AND (typhoon OR storm)"
            
            url = "https://newsapi.org/v2/everything"
            params = {
                'q': query,
                'from': from_date.strftime('%Y-%m-%d'),
                'to': to_date.strftime('%Y-%m-%d'),
                'sortBy': 'publishedAt',
                'pageSize': 10,
                'apiKey': self.newsapi_key
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            articles = data.get('articles', [])
            
            # Filter and process articles
            filtered_articles = []
            for article in articles:
                if self._is_valid_article(article):
                    filtered_articles.append(article)
            
            return filtered_articles[:10]  # Limit to 10 articles
            
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Error fetching news: {str(e)}")
            return []
        except Exception as e:
            st.error(f"❌ Unexpected error: {str(e)}")
            return []
    
    def _is_real_estate_stock(self, symbol: str) -> bool:
        """Check if stock is in real estate sector"""
        # Hong Kong real estate stock patterns
        real_estate_patterns = [
            r'\.HK$',  # Hong Kong stocks
            r'^0[0-9]{3}\.HK$',  # HKEX real estate codes
        ]
        
        # Common Hong Kong real estate stock symbols
        real_estate_symbols = [
            '0001.HK', '0002.HK', '0003.HK', '0004.HK', '0005.HK',
            '0016.HK', '0017.HK', '0019.HK', '0023.HK', '0027.HK',
            '0069.HK', '0083.HK', '0101.HK', '0113.HK', '0116.HK',
            '0127.HK', '0135.HK', '0138.HK', '0142.HK', '0151.HK'
        ]
        
        return symbol.upper() in real_estate_symbols or any(
            re.match(pattern, symbol.upper()) for pattern in real_estate_patterns
        )
    
    def _is_valid_article(self, article: Dict) -> bool:
        """Validate article content and language"""
        try:
            title = article.get('title', '')
            description = article.get('description', '')
            content = f"{title} {description}"
            
            if not content.strip():
                return False
            
            # Detect language
            language = detect(content)
            return language in ['en', 'zh-cn', 'zh-tw']
            
        except LangDetectException:
            return False
        except Exception:
            return False
    
    def detect_language(self, text: str) -> str:
        """Detect language of text"""
        try:
            lang = detect(text)
            if lang in ['zh-cn', 'zh-tw']:
                return 'zh'
            return 'en'
        except LangDetectException:
            return 'en'  # Default to English
    
    def analyze_sentiment(self, text: str, language: str) -> Tuple[float, str]:
        """Analyze sentiment of text"""
        try:
            if language == 'zh':
                # Use SnowNLP for Chinese
                s = SnowNLP(text)
                score = s.sentiments
                if score > 0.6:
                    sentiment = "positive"
                elif score < 0.4:
                    sentiment = "negative"
                else:
                    sentiment = "neutral"
            else:
                # Use TextBlob for English
                blob = TextBlob(text)
                score = blob.sentiment.polarity
                if score > 0.1:
                    sentiment = "positive"
                elif score < -0.1:
                    sentiment = "negative"
                else:
                    sentiment = "neutral"
            
            return score, sentiment
            
        except Exception as e:
            st.warning(f"Sentiment analysis failed: {str(e)}")
            return 0.0, "neutral"
    
    def summarize_news(self, articles: List[Dict]) -> str:
        """Generate AI summary of all news articles"""
        if not self.summarizer or not articles:
            return "No summary available."
        
        try:
            # Combine all article content
            combined_text = ""
            for article in articles:
                title = article.get('title', '')
                description = article.get('description', '')
                content = f"{title}. {description}"
                combined_text += content + " "
            
            # Truncate if too long (BART has token limits)
            if len(combined_text) > 1000:
                combined_text = combined_text[:1000]
            
            # Generate summary
            summary = self.summarizer(
                combined_text,
                max_length=150,
                min_length=50,
                do_sample=False
            )
            
            return summary[0]['summary_text']
            
        except Exception as e:
            st.error(f"❌ Summarization failed: {str(e)}")
            return "Summary generation failed."

def main():
    # Header
    st.markdown('<h1 class="main-header">📈 Hong Kong FIRE Stock News Summarizer</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 🏢 Hong Kong FIRE Stock Tool")
        st.markdown("**2025 Version**")
        st.markdown("---")
        st.markdown("### 📊 Features")
        st.markdown("- 📰 Latest news fetching")
        st.markdown("- 🤖 AI-powered summarization")
        st.markdown("- 😊 Sentiment analysis")
        st.markdown("- 📈 Data visualization")
        st.markdown("- 🌪️ Typhoon risk alerts")
        st.markdown("---")
        st.markdown("### 🎯 Target Industries")
        st.markdown("- **F**inance")
        st.markdown("- **I**nsurance") 
        st.markdown("- **R**eal **E**state")
        st.markdown("---")
        st.markdown("### 💡 Usage")
        st.markdown("Enter a stock symbol (e.g., AAPL, 0700.HK) to get started!")
    
    # Initialize the summarizer
    if 'summarizer' not in st.session_state:
        st.session_state.summarizer = StockNewsSummarizer()
    
    # Main input
    col1, col2 = st.columns([2, 1])
    
    with col1:
        symbol = st.text_input(
            "📈 Enter Stock Symbol",
            placeholder="e.g., AAPL, 0700.HK, 0001.HK",
            help="Enter a stock symbol. For Hong Kong stocks, use format like 0700.HK"
        ).upper()
    
    with col2:
        days_back = st.selectbox(
            "📅 News Period",
            options=[1, 3, 7, 14, 30],
            index=2,
            help="Number of days to look back for news"
        )
    
    if st.button("🔍 Analyze Stock News", type="primary"):
        if not symbol:
            st.error("❌ Please enter a stock symbol")
            return
        
        with st.spinner("🔄 Fetching and analyzing news..."):
            # Fetch news
            articles = st.session_state.summarizer.get_news(symbol, days_back)
            
            if not articles:
                st.warning("⚠️ No news found for this symbol. Try a different symbol or time period.")
                return
            
            # Process articles
            processed_articles = []
            sentiments = []
            
            for article in articles:
                title = article.get('title', '')
                description = article.get('description', '')
                url = article.get('url', '')
                published_at = article.get('publishedAt', '')
                
                # Detect language
                content = f"{title} {description}"
                language = st.session_state.summarizer.detect_language(content)
                
                # Analyze sentiment
                sentiment_score, sentiment_label = st.session_state.summarizer.analyze_sentiment(
                    content, language
                )
                
                processed_articles.append({
                    'title': title,
                    'description': description,
                    'url': url,
                    'published_at': published_at,
                    'language': language,
                    'sentiment_score': sentiment_score,
                    'sentiment_label': sentiment_label
                })
                
                sentiments.append(sentiment_label)
            
            # Display results
            st.success(f"✅ Found {len(processed_articles)} news articles for {symbol}")
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("📰 Total Articles", len(processed_articles))
            
            with col2:
                positive_count = sentiments.count('positive')
                st.metric("😊 Positive", positive_count)
            
            with col3:
                negative_count = sentiments.count('negative')
                st.metric("😞 Negative", negative_count)
            
            with col4:
                neutral_count = sentiments.count('neutral')
                st.metric("😐 Neutral", neutral_count)
            
            # AI Summary
            st.markdown("## 🤖 AI Summary")
            summary = st.session_state.summarizer.summarize_news(articles)
            st.info(summary)
            
            # Sentiment Visualization
            st.markdown("## 📊 Sentiment Distribution")
            
            sentiment_counts = pd.DataFrame({
                'Sentiment': ['Positive', 'Negative', 'Neutral'],
                'Count': [
                    sentiments.count('positive'),
                    sentiments.count('negative'),
                    sentiments.count('neutral')
                ]
            })
            
            fig = px.bar(
                sentiment_counts,
                x='Sentiment',
                y='Count',
                color='Sentiment',
                color_discrete_map={
                    'Positive': '#28a745',
                    'Negative': '#dc3545',
                    'Neutral': '#6c757d'
                }
            )
            fig.update_layout(
                title="News Sentiment Distribution",
                xaxis_title="Sentiment",
                yaxis_title="Number of Articles",
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Overall Insights
            st.markdown("## 💡 Overall Insights")
            
            col1, col2 = st.columns(2)
            
            with col1:
                avg_sentiment = sum([a['sentiment_score'] for a in processed_articles]) / len(processed_articles)
                st.metric("📈 Average Sentiment Score", f"{avg_sentiment:.3f}")
            
            with col2:
                if st.session_state.summarizer._is_real_estate_stock(symbol):
                    st.metric("🌪️ Typhoon Risk", "Monitored", help="Real estate stocks include typhoon/storm news")
                else:
                    st.metric("🌪️ Typhoon Risk", "Not Applicable")
            
            # News List
            st.markdown("## 📰 News Articles")
            
            for i, article in enumerate(processed_articles, 1):
                with st.expander(f"📄 Article {i}: {article['title'][:60]}..."):
                    st.markdown(f"**Title:** {article['title']}")
                    st.markdown(f"**Description:** {article['description']}")
                    st.markdown(f"**Published:** {article['published_at']}")
                    st.markdown(f"**Language:** {article['language'].upper()}")
                    
                    # Sentiment display
                    sentiment_class = f"sentiment-{article['sentiment_label']}"
                    st.markdown(f"**Sentiment:** <span class='{sentiment_class}'>{article['sentiment_label'].title()}</span> (Score: {article['sentiment_score']:.3f})", unsafe_allow_html=True)
                    
                    if article['url']:
                        st.markdown(f"**Link:** [Read Full Article]({article['url']})")

if __name__ == "__main__":
    main()
