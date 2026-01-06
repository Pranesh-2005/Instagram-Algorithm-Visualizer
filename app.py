import gradio as gr
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import math
import random
from datetime import datetime, timedelta
import json

# ---------------------------
# GLOBAL DATA (SIMULATION)
# ---------------------------

TOPICS = ["AI/Tech", "Music", "Travel", "Food", "Gaming", "Fitness", "Fashion", "Art", "Business", "Comedy"]

# More complex global trends with temporal variations
GLOBAL_TRENDING = {
    "AI/Tech": {"base": 0.95, "volatility": 0.1, "seasonal": 1.2},
    "Music": {"base": 0.90, "volatility": 0.15, "seasonal": 1.1},
    "Travel": {"base": 0.88, "volatility": 0.2, "seasonal": 0.8},
    "Food": {"base": 0.85, "volatility": 0.05, "seasonal": 1.0},
    "Gaming": {"base": 0.82, "volatility": 0.12, "seasonal": 1.0},
    "Fitness": {"base": 0.80, "volatility": 0.08, "seasonal": 1.3},
    "Fashion": {"base": 0.87, "volatility": 0.18, "seasonal": 1.1},
    "Art": {"base": 0.75, "volatility": 0.10, "seasonal": 0.9},
    "Business": {"base": 0.78, "volatility": 0.07, "seasonal": 0.95},
    "Comedy": {"base": 0.92, "volatility": 0.20, "seasonal": 1.0}
}

# Content creator tiers
CREATOR_TIERS = {
    "Micro": {"followers": 1000, "engagement_mult": 1.2, "reach_mult": 0.8},
    "Mid": {"followers": 50000, "engagement_mult": 1.0, "reach_mult": 1.0},
    "Macro": {"followers": 500000, "engagement_mult": 0.8, "reach_mult": 1.5},
    "Celebrity": {"followers": 5000000, "engagement_mult": 0.6, "reach_mult": 2.0}
}

# User demographics
DEMOGRAPHICS = ["Gen Z", "Millennial", "Gen X", "Boomer"]
REGIONS = ["North America", "Europe", "Asia", "South America", "Africa"]

np.random.seed(42)

# More complex embedding space (higher dimensional)
EMBEDDING_DIM = 8
POST_EMBEDDINGS = {}
USER_DEMOGRAPHICS = {}

for topic in TOPICS:
    base_trend = GLOBAL_TRENDING[topic]["base"]
    volatility = GLOBAL_TRENDING[topic]["volatility"]
    embedding = np.random.randn(EMBEDDING_DIM) * base_trend + np.random.randn(EMBEDDING_DIM) * volatility
    POST_EMBEDDINGS[topic] = embedding

GLOBAL_EMBEDDING = np.mean(list(POST_EMBEDDINGS.values()), axis=0)

# ---------------------------
# ENHANCED UTILS
# ---------------------------

def sigmoid(x, steepness=1.0):
    return 1 / (1 + math.exp(-steepness * x))

def attention_mechanism(query, keys, values, temperature=1.0):
    """Simplified attention mechanism for content ranking"""
    scores = np.dot(keys, query) / temperature
    weights = np.exp(scores) / np.sum(np.exp(scores))
    return np.dot(weights, values), weights

def temporal_decay(time_diff_hours, half_life=24):
    """Content freshness decay"""
    return 0.5 ** (time_diff_hours / half_life)

def diversity_penalty(selected_topics, candidate_topic, penalty_strength=0.1):
    """Penalty for showing too much of the same content"""
    count = selected_topics.count(candidate_topic)
    return math.exp(-penalty_strength * count)

# ---------------------------
# ENHANCED VALUE MODEL
# ---------------------------

def enhanced_value_model(like_signal, comment_signal, share_signal, save_signal, 
                        watch_time, creator_tier, recency_hours, user_history_match):
    """More sophisticated value model with multiple signals"""
    
    # Base probabilities
    p_like = sigmoid(like_signal - 1, steepness=0.8)
    p_comment = sigmoid(comment_signal - 0.5, steepness=1.2)
    p_share = sigmoid(share_signal - 0.3, steepness=1.5)
    p_save = sigmoid(save_signal - 0.2, steepness=1.0)
    p_watch = sigmoid(watch_time - 2, steepness=0.6)
    
    # Creator influence
    creator_mult = CREATOR_TIERS[creator_tier]["engagement_mult"]
    
    # Recency factor
    recency_factor = temporal_decay(recency_hours)
    
    # User history matching
    history_boost = sigmoid(user_history_match - 0.5, steepness=2.0)
    
    # Weighted scoring
    base_score = (
        1.0 * p_like +
        3.0 * p_comment +
        4.0 * p_share +
        2.5 * p_save +
        1.5 * p_watch
    )
    
    # Apply modifiers
    final_score = base_score * creator_mult * recency_factor * (0.5 + 0.5 * history_boost)
    
    return round(final_score, 4), {
        "P(Like)": round(p_like, 3),
        "P(Comment)": round(p_comment, 3),
        "P(Share)": round(p_share, 3),
        "P(Save)": round(p_save, 3),
        "P(Watch)": round(p_watch, 3),
        "Creator Multiplier": round(creator_mult, 3),
        "Recency Factor": round(recency_factor, 3),
        "History Match": round(history_boost, 3)
    }

# ---------------------------
# ENHANCED COLD START WITH EXPLORATION
# ---------------------------

def enhanced_cold_start(interactions, preferred_topics, demographics, region, 
                       exploration_factor=0.2):
    """Enhanced cold start with demographic targeting and exploration"""
    
    # Blending factor based on interactions
    alpha = min(interactions / 50.0, 0.9)  # More gradual transition
    
    # Create personal embedding from multiple preferences
    if len(preferred_topics) > 0:
        personal_embeddings = [POST_EMBEDDINGS[topic] for topic in preferred_topics]
        personal_embedding = np.mean(personal_embeddings, axis=0)
    else:
        personal_embedding = GLOBAL_EMBEDDING
    
    # Demographic influence
    demo_noise = np.random.randn(EMBEDDING_DIM) * 0.1
    if demographics == "Gen Z":
        demo_noise += np.array([0.2, -0.1, 0.3, 0.1, 0.2, -0.1, 0.1, 0.2])
    elif demographics == "Millennial":
        demo_noise += np.array([0.1, 0.2, 0.1, 0.2, 0.1, 0.2, 0.1, 0.0])
    
    # Regional trends
    regional_bias = np.zeros(EMBEDDING_DIM)
    if region == "Asia":
        regional_bias += np.array([0.3, 0.1, -0.2, 0.2, 0.1, 0.0, 0.1, 0.1])
    elif region == "North America":
        regional_bias += np.array([0.1, 0.2, 0.1, 0.1, 0.3, 0.1, 0.2, 0.1])
    
    # Exploration component (random discovery)
    exploration_noise = np.random.randn(EMBEDDING_DIM) * exploration_factor
    
    # Final blended embedding
    user_embedding = (
        alpha * personal_embedding + 
        (1 - alpha) * GLOBAL_EMBEDDING + 
        demo_noise + 
        regional_bias + 
        exploration_noise
    )
    
    return user_embedding

# ---------------------------
# CONTENT RANKING SYSTEM
# ---------------------------

def rank_content_feed(user_embedding, content_pool_size=20, diversity_weight=0.3):
    """Simulate full feed ranking with diversity considerations"""
    
    # Generate synthetic content
    content_items = []
    for i in range(content_pool_size):
        topic = random.choice(TOPICS)
        creator_tier = random.choices(
            list(CREATOR_TIERS.keys()), 
            weights=[40, 35, 20, 5]
        )[0]
        
        # Content embedding with some noise
        content_emb = POST_EMBEDDINGS[topic] + np.random.randn(EMBEDDING_DIM) * 0.1
        
        # Relevance score (cosine similarity)
        relevance = np.dot(user_embedding, content_emb) / (
            np.linalg.norm(user_embedding) * np.linalg.norm(content_emb)
        )
        
        # Random engagement signals
        engagement_signals = {
            "likes": max(0, np.random.normal(5, 2)),
            "comments": max(0, np.random.normal(2, 1)),
            "shares": max(0, np.random.normal(1, 0.5)),
            "saves": max(0, np.random.normal(0.8, 0.3)),
            "watch_time": max(0, np.random.normal(4, 1.5)),
            "recency": np.random.uniform(0.1, 48)
        }
        
        # Calculate value score
        value_score, _ = enhanced_value_model(
            engagement_signals["likes"],
            engagement_signals["comments"], 
            engagement_signals["shares"],
            engagement_signals["saves"],
            engagement_signals["watch_time"],
            creator_tier,
            engagement_signals["recency"],
            max(0, relevance)
        )
        
        content_items.append({
            "id": i,
            "topic": topic,
            "creator_tier": creator_tier,
            "relevance": relevance,
            "value_score": value_score,
            "embedding": content_emb,
            **engagement_signals
        })
    
    # Rank with diversity
    ranked_items = []
    remaining_items = content_items.copy()
    selected_topics = []
    
    for position in range(min(10, len(remaining_items))):
        best_item = None
        best_score = -float('inf')
        
        for item in remaining_items:
            # Combined score: relevance + diversity
            diversity_score = diversity_penalty(selected_topics, item["topic"])
            combined_score = (
                (1 - diversity_weight) * item["value_score"] + 
                diversity_weight * diversity_score
            )
            
            if combined_score > best_score:
                best_score = combined_score
                best_item = item
        
        if best_item:
            ranked_items.append(best_item)
            selected_topics.append(best_item["topic"])
            remaining_items.remove(best_item)
    
    return ranked_items

# ---------------------------
# ENHANCED UI FUNCTIONS
# ---------------------------

def tab_enhanced_value_model(likes, comments, shares, saves, watch_time, creator_tier, recency, history_match):
    score, metrics = enhanced_value_model(
        likes, comments, shares, saves, watch_time, creator_tier, recency, history_match
    )

    # Create metrics visualization
    fig = go.Figure()
    
    metric_names = list(metrics.keys())[:5]  # First 5 are probabilities
    metric_values = [metrics[name] for name in metric_names]
    
    fig.add_trace(go.Bar(
        x=metric_names,
        y=metric_values,
        marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57']
    ))
    
    fig.update_layout(
        title="Signal Probabilities",
        yaxis_title="Probability",
        height=400
    )

    text = f"""
### 🔢 Enhanced Value Model Output

**Final Ranking Score: {score}**

#### 📊 Signal Probabilities:
- 👍 Like: **{metrics['P(Like)']}**
- 💬 Comment: **{metrics['P(Comment)']}**
- 🔄 Share: **{metrics['P(Share)']}**
- 📌 Save: **{metrics['P(Save)']}**
- 👀 Watch: **{metrics['P(Watch)']}**

#### 🎯 Modifiers:
- Creator Influence: **{metrics['Creator Multiplier']}**
- Content Freshness: **{metrics['Recency Factor']}**
- User History Match: **{metrics['History Match']}**
"""

    return text, fig

def tab_enhanced_cold_start(interactions, topics, demographics, region, exploration):
    user_vec = enhanced_cold_start(interactions, topics, demographics, region, exploration)
    
    # Calculate distances to different content types
    distances = {}
    for topic, embedding in POST_EMBEDDINGS.items():
        distance = np.linalg.norm(user_vec - embedding)
        distances[topic] = distance
    
    # Sort by proximity
    sorted_topics = sorted(distances.items(), key=lambda x: x[1])
    
    explanation = f"""
### 🧊 Enhanced Cold Start Analysis

**User Profile:**
- Interactions: **{interactions}**
- Demographics: **{demographics}** in **{region}**
- Exploration Factor: **{exploration}**
- Preferred Topics: **{', '.join(topics) if topics else 'None selected'}**

#### 🎯 Content Affinity (Closest → Farthest):
"""
    
    for i, (topic, dist) in enumerate(sorted_topics[:5]):
        explanation += f"{i+1}. **{topic}** (distance: {dist:.3f})\n"

    return explanation, user_vec

def tab_feed_ranking(user_vec):
    if user_vec is None:
        return "Please generate a user profile first in the Cold Start tab.", None
    
    ranked_content = rank_content_feed(user_vec)
    
    # Create feed visualization
    df_feed = pd.DataFrame([
        {
            "Position": i+1,
            "Topic": item["topic"],
            "Creator": item["creator_tier"],
            "Relevance": round(item["relevance"], 3),
            "Value Score": round(item["value_score"], 3),
            "Likes": round(item["likes"], 1),
            "Comments": round(item["comments"], 1),
            "Shares": round(item["shares"], 1)
        }
        for i, item in enumerate(ranked_content)
    ])
    
    # Create ranking visualization
    fig = px.scatter(
        df_feed,
        x="Relevance",
        y="Value Score", 
        size="Likes",
        color="Topic",
        hover_data=["Creator", "Comments", "Shares"],
        title="Content Ranking: Relevance vs Value Score"
    )
    
    return df_feed, fig

def tab_advanced_analytics(user_vec):
    if user_vec is None:
        return None, None, "Generate user profile first"
    
    # Topic affinity radar chart
    topic_scores = []
    for topic, embedding in POST_EMBEDDINGS.items():
        similarity = np.dot(user_vec, embedding) / (
            np.linalg.norm(user_vec) * np.linalg.norm(embedding)
        )
        topic_scores.append(similarity)
    
    fig_radar = go.Figure()
    
    fig_radar.add_trace(go.Scatterpolar(
        r=topic_scores,
        theta=TOPICS,
        fill='toself',
        name='User Affinity'
    ))
    
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[-1, 1]
            )),
        showlegend=True,
        title="User Topic Affinity Profile"
    )
    
    # Embedding visualization (PCA to 2D)
    all_embeddings = list(POST_EMBEDDINGS.values()) + [user_vec]
    all_labels = TOPICS + ["User"]
    
    # Simple 2D projection (first two dimensions)
    x_coords = [emb[0] for emb in all_embeddings]
    y_coords = [emb[1] for emb in all_embeddings]
    
    fig_embed = go.Figure()
    
    # Plot topics
    for i, (x, y, label) in enumerate(zip(x_coords[:-1], y_coords[:-1], all_labels[:-1])):
        fig_embed.add_trace(go.Scatter(
            x=[x], y=[y],
            mode='markers+text',
            text=[label],
            textposition="top center",
            marker=dict(size=10),
            name=label
        ))
    
    # Plot user
    fig_embed.add_trace(go.Scatter(
        x=[x_coords[-1]], y=[y_coords[-1]],
        mode='markers+text',
        text=["You"],
        textposition="top center",
        marker=dict(size=15, color='red'),
        name="User"
    ))
    
    fig_embed.update_layout(
        title="2D Embedding Space Projection",
        xaxis_title="Dimension 1",
        yaxis_title="Dimension 2"
    )
    
    # Statistics
    stats = f"""
### 📈 Advanced Analytics

**User Vector Statistics:**
- Vector Magnitude: {np.linalg.norm(user_vec):.3f}
- Dominant Dimensions: {np.argmax(np.abs(user_vec))}, {np.argsort(np.abs(user_vec))[-2]}
- Diversity Score: {np.std(topic_scores):.3f}
- Global Alignment: {np.dot(user_vec, GLOBAL_EMBEDDING) / (np.linalg.norm(user_vec) * np.linalg.norm(GLOBAL_EMBEDDING)):.3f}
"""
    
    return fig_radar, fig_embed, stats

# ---------------------------
# ENHANCED GRADIO UI
# ---------------------------

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 📸 Advanced Instagram Recommendation Algorithm Simulator")
    gr.Markdown("### Explore the complex mechanics behind social media content ranking")

    with gr.Tabs():

        # ---------------- TAB A: Enhanced Value Model ----------------
        with gr.Tab("🔢 Value Model"):
            gr.Markdown("### Multi-Signal Content Scoring System")
            
            with gr.Row():
                with gr.Column():
                    likes = gr.Slider(0, 20, 5, label="👍 Likes Signal")
                    comments = gr.Slider(0, 10, 2, label="💬 Comments Signal")
                    shares = gr.Slider(0, 5, 1, label="🔄 Shares Signal")
                    saves = gr.Slider(0, 3, 0.5, label="📌 Saves Signal")
                    
                with gr.Column():
                    watch_time = gr.Slider(0, 15, 5, label="👀 Watch Time (seconds)")
                    creator_tier = gr.Dropdown(list(CREATOR_TIERS.keys()), value="Mid", label="👤 Creator Tier")
                    recency = gr.Slider(0.1, 48, 2, label="⏰ Hours Since Posted")
                    history_match = gr.Slider(0, 1, 0.5, label="🎯 Based on your history %")

            value_output = gr.Markdown()
            value_chart = gr.Plot()

            gr.Button("🚀 Calculate Ranking Score", variant="primary").click(
                tab_enhanced_value_model,
                inputs=[likes, comments, shares, saves, watch_time, creator_tier, recency, history_match],
                outputs=[value_output, value_chart]
            )

        # ---------------- TAB B: Enhanced Cold Start ----------------
        with gr.Tab("🧊 Cold Start & Personalization"):
            gr.Markdown("### From Generic → Personalized Content")
            
            with gr.Row():
                with gr.Column():
                    interactions = gr.Slider(0, 100, 0, label="📱 Total User Interactions")
                    topics = gr.CheckboxGroup(TOPICS, label="❤️ Preferred Topics")
                    
                with gr.Column():
                    demographics = gr.Dropdown(DEMOGRAPHICS, value="Gen Z", label="👥 Demographics")
                    region = gr.Dropdown(REGIONS, value="North America", label="🌍 Region")
                    exploration = gr.Slider(0, 0.5, 0.2, label="🎲 New activity level")

            cold_start_output = gr.Markdown()
            user_vec_state = gr.State()

            gr.Button("🎭 Generate User Profile", variant="primary").click(
                tab_enhanced_cold_start,
                inputs=[interactions, topics, demographics, region, exploration],
                outputs=[cold_start_output, user_vec_state]
            )

        # ---------------- TAB C: Feed Ranking ----------------
        with gr.Tab("📱 Feed Ranking Simulation"):
            gr.Markdown("### See Your Personalized Feed in Action")

            feed_table = gr.Dataframe()
            feed_chart = gr.Plot()

            gr.Button("🔄 Generate My Feed", variant="primary").click(
                tab_feed_ranking,
                inputs=[user_vec_state],
                outputs=[feed_table, feed_chart]
            )

        # ---------------- TAB D: Advanced Analytics ----------------
        with gr.Tab("📊 Advanced Analytics"):
            gr.Markdown("### Deep Dive into Algorithm Mechanics")

            analytics_stats = gr.Markdown()
            
            with gr.Row():
                radar_chart = gr.Plot()
                embedding_chart = gr.Plot()

            gr.Button("🔬 Analyze User Profile", variant="primary").click(
                tab_advanced_analytics,
                inputs=[user_vec_state],
                outputs=[radar_chart, embedding_chart, analytics_stats]
            )

    # ---------------- Information Panel ----------------
    with gr.Accordion("📚 Algorithm Insights", open=False):
        gr.Markdown("""
### How This Simulation Works:

1. **Value Model**: Converts user engagement signals into probability scores using sigmoid functions
2. **Cold Start**: Blends global trends with personal preferences based on interaction history
3. **Embeddings**: Represents users and content in high-dimensional vector space
4. **Ranking**: Combines relevance scores with diversity penalties for balanced feeds
5. **Personalization**: Gradually shifts from trending to personalized content

### Key Concepts:
- **Attention Mechanism**: Weighted content selection based on user interests
- **Temporal Decay**: Newer content gets priority boost
- **Diversity Penalty**: Prevents echo chambers by promoting content variety
- **Demographic Targeting**: Adjusts recommendations based on user demographics
- **Exploration vs Exploitation**: Balance between showing familiar and new content
""")

    with gr.Accordion("⚙️ Technical Implementation", open=False):
        gr.Markdown("""
### Advanced Features:

- **8-Dimensional Embedding Space** for richer content representation
- **Multi-Signal Value Model** with 5 engagement types
- **Demographic & Regional Biases** in recommendation
- **Dynamic Exploration Factor** for content discovery  
- **Attention-Based Ranking** with diversity constraints
- **Temporal Content Decay** for freshness prioritization
- **Creator Tier Influence** on engagement predictions
""")

if __name__ == "__main__":
    demo.launch(share=True, debug=True)