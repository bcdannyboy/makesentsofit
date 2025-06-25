"""
Advanced User-Sentiment Network Analysis
Maps users who talk most about topics with positive/negative sentiment patterns.
"""
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict, Counter
import math


class UserSentimentNetworkAnalyzer:
    """Analyze and visualize user sentiment networks and activity patterns."""
    
    def __init__(self):
        self.sentiment_colors = {
            'POSITIVE': '#28a745',  # Green
            'NEGATIVE': '#dc3545',  # Red  
            'NEUTRAL': '#6c757d',   # Gray
            'MIXED': '#ffc107'      # Yellow
        }
        
    def analyze_user_patterns(self, posts: List["Post"]) -> Dict:
        """Analyze user posting patterns and sentiment distribution."""
        if not posts:
            return {}
            
        user_stats = defaultdict(lambda: {
            'post_count': 0,
            'sentiments': Counter(),
            'topics': Counter(),
            'subreddits': Counter(),
            'total_engagement': 0,
            'avg_engagement': 0,
            'posts': []
        })
        
        # Aggregate user data
        for post in posts:
            if not hasattr(post, 'sentiment') or not post.sentiment:
                continue
                
            user = post.author
            sentiment = post.sentiment.get('label', 'NEUTRAL')
            query = getattr(post, 'query', 'unknown')
            engagement = sum(post.engagement.values()) if post.engagement else 0
            subreddit = post.metadata.get('subreddit')

            user_stats[user]['post_count'] += 1
            user_stats[user]['sentiments'][sentiment] += 1
            user_stats[user]['topics'][query] += 1
            if subreddit:
                user_stats[user]['subreddits'][subreddit] += 1
            user_stats[user]['total_engagement'] += engagement
            user_stats[user]['posts'].append(post)
        
        # Calculate derived metrics
        for user, stats in user_stats.items():
            if stats['post_count'] > 0:
                stats['avg_engagement'] = stats['total_engagement'] / stats['post_count']
                
                # Determine dominant sentiment
                most_common_sentiment = stats['sentiments'].most_common(1)
                if most_common_sentiment:
                    dominant_sentiment, count = most_common_sentiment[0]
                    sentiment_ratio = count / stats['post_count']
                    
                    # Classify user sentiment pattern
                    if sentiment_ratio >= 0.7:
                        stats['user_type'] = dominant_sentiment
                    elif sentiment_ratio >= 0.4:
                        stats['user_type'] = 'MIXED'
                    else:
                        stats['user_type'] = 'NEUTRAL'
                else:
                    stats['user_type'] = 'NEUTRAL'
                    
                # Calculate sentiment scores
                pos_count = stats['sentiments']['POSITIVE']
                neg_count = stats['sentiments']['NEGATIVE']
                neu_count = stats['sentiments']['NEUTRAL']
                
                if stats['post_count'] > 0:
                    stats['sentiment_score'] = (pos_count - neg_count) / stats['post_count']
                    stats['positivity_ratio'] = pos_count / stats['post_count']
                    stats['negativity_ratio'] = neg_count / stats['post_count']
                else:
                    stats['sentiment_score'] = 0
                    stats['positivity_ratio'] = 0
                    stats['negativity_ratio'] = 0
        
        return dict(user_stats)
    
    def create_user_activity_network(self, posts: List["Post"], min_posts: int = 3) -> nx.Graph:
        """Create network graph of user interactions and shared topics."""
        user_stats = self.analyze_user_patterns(posts)
        
        # Filter active users
        active_users = {
            user: stats for user, stats in user_stats.items() 
            if stats['post_count'] >= min_posts
        }
        
        G = nx.Graph()
        
        # Add user nodes with attributes
        for user, stats in active_users.items():
            G.add_node(user, **{
                'post_count': stats['post_count'],
                'user_type': stats['user_type'],
                'sentiment_score': stats['sentiment_score'],
                'avg_engagement': stats['avg_engagement'],
                'total_engagement': stats['total_engagement'],
                'positivity_ratio': stats['positivity_ratio'],
                'negativity_ratio': stats['negativity_ratio'],
                'top_topics': list(stats['topics'].most_common(3)),
                'top_subreddits': list(stats['subreddits'].most_common(3))
            })
        
        # Create edges based on shared topics and similar sentiment patterns
        users = list(active_users.keys())
        for i, user1 in enumerate(users):
            for user2 in users[i+1:]:
                # Calculate topic overlap
                topics1 = set(active_users[user1]['topics'].keys())
                topics2 = set(active_users[user2]['topics'].keys())
                shared_topics = topics1.intersection(topics2)

                sub1 = set(active_users[user1]['subreddits'].keys())
                sub2 = set(active_users[user2]['subreddits'].keys())
                shared_subs = sub1.intersection(sub2)

                if shared_topics or shared_subs:
                    topic_overlap = len(shared_topics) / len(topics1.union(topics2)) if topics1.union(topics2) else 0
                    sub_overlap = len(shared_subs) / len(sub1.union(sub2)) if sub1.union(sub2) else 0

                    score1 = active_users[user1]['sentiment_score']
                    score2 = active_users[user2]['sentiment_score']
                    sentiment_similarity = 1 - abs(score1 - score2)

                    weight = (topic_overlap + sub_overlap + sentiment_similarity) / 3

                    if weight > 0.3:
                        G.add_edge(
                            user1,
                            user2,
                            weight=weight,
                            shared_topics=list(shared_topics),
                            shared_subreddits=list(shared_subs),
                            topic_overlap=topic_overlap,
                            subreddit_overlap=sub_overlap,
                            sentiment_similarity=sentiment_similarity,
                        )
        
        return G
    
    def create_plotly_network_viz(self, posts: List["Post"], output_path: str, 
                                 min_posts: int = 3) -> None:
        """Create interactive Plotly network visualization."""
        G = self.create_user_activity_network(posts, min_posts)
        
        if len(G.nodes()) == 0:
            return
        
        # Calculate layout
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Prepare node data
        node_x = []
        node_y = []
        node_text = []
        node_colors = []
        node_sizes = []
        node_info = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            # Node attributes
            attrs = G.nodes[node]
            user_type = attrs['user_type']
            post_count = attrs['post_count']
            sentiment_score = attrs['sentiment_score']
            avg_engagement = attrs['avg_engagement']
            top_topics = attrs.get('top_topics', [])
            top_subs = attrs.get('top_subreddits', [])
            
            # Color by sentiment type
            node_colors.append(self.sentiment_colors.get(user_type, '#6c757d'))
            
            # Size by activity level (post count)
            size = max(10, min(50, post_count * 3))
            node_sizes.append(size)
            
            # Hover text
            topics_str = ', '.join([f"{topic} ({count})" for topic, count in top_topics[:3]])
            subs_str = ', '.join([f"r/{sub} ({count})" for sub, count in top_subs[:3]])
            hover_text = (
                f"<b>{node}</b><br>"
                f"Posts: {post_count}<br>"
                f"Sentiment: {user_type}<br>"
                f"Score: {sentiment_score:.2f}<br>"
                f"Avg Engagement: {avg_engagement:.1f}<br>"
                f"Top Topics: {topics_str}<br>"
                f"Subreddits: {subs_str}"
            )
            node_info.append(hover_text)
            node_text.append(node)
        
        # Prepare edge data
        edge_x = []
        edge_y = []
        edge_weights = []
        
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_weights.append(G.edges[edge]['weight'])
        
        # Create traces
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='rgba(125,125,125,0.5)'),
            hoverinfo='none',
            mode='lines'
        )
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            hovertext=node_info,
            text=[t if len(t) < 10 else t[:7]+'...' for t in node_text],
            textposition="middle center",
            textfont=dict(size=8),
            marker=dict(
                size=node_sizes,
                color=node_colors,
                line=dict(width=2, color='white'),
                opacity=0.8
            )
        )
        
        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title=dict(
                               text="User Sentiment Network - Activity & Opinion Leaders",
                               x=0.5,
                               font=dict(size=16)
                           ),
                           titlefont_size=16,
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=40),
                           annotations=[ dict(
                               text="Node size = post count | Color = dominant sentiment | Lines = shared topics",
                               showarrow=False,
                               xref="paper", yref="paper",
                               x=0.005, y=-0.002,
                               xanchor='left', yanchor='bottom',
                               font=dict(color='gray', size=10)
                           )],
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           plot_bgcolor='white'
                       ))
        
        fig.write_html(output_path)
    
    def create_sentiment_leaders_analysis(self, posts: List["Post"], output_path: str,
                                        top_n: int = 20) -> None:
        """Create analysis of top positive and negative sentiment leaders."""
        user_stats = self.analyze_user_patterns(posts)
        
        # Filter users with minimum activity
        active_users = {
            user: stats for user, stats in user_stats.items() 
            if stats['post_count'] >= 2
        }
        
        if not active_users:
            return
        
        # Sort by different metrics
        by_posts = sorted(active_users.items(), key=lambda x: x[1]['post_count'], reverse=True)
        by_positive = sorted(active_users.items(), key=lambda x: x[1]['positivity_ratio'], reverse=True)
        by_negative = sorted(active_users.items(), key=lambda x: x[1]['negativity_ratio'], reverse=True)
        by_engagement = sorted(active_users.items(), key=lambda x: x[1]['avg_engagement'], reverse=True)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Most Active Users (by post count)',
                'Most Positive Users (by ratio)',
                'Most Negative Users (by ratio)', 
                'Highest Engagement Users'
            ],
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Most active users
        top_active = by_posts[:top_n]
        fig.add_trace(
            go.Bar(
                x=[user for user, _ in top_active],
                y=[stats['post_count'] for _, stats in top_active],
                name='Posts',
                marker_color='lightblue'
            ),
            row=1, col=1
        )
        
        # Most positive users (filter for meaningful ratios)
        pos_users = [(u, s) for u, s in by_positive if s['post_count'] >= 3 and s['positivity_ratio'] > 0.5][:top_n]
        fig.add_trace(
            go.Bar(
                x=[user for user, _ in pos_users],
                y=[stats['positivity_ratio'] for _, stats in pos_users],
                name='Positivity',
                marker_color='green'
            ),
            row=1, col=2
        )
        
        # Most negative users
        neg_users = [(u, s) for u, s in by_negative if s['post_count'] >= 3 and s['negativity_ratio'] > 0.5][:top_n]
        fig.add_trace(
            go.Bar(
                x=[user for user, _ in neg_users],
                y=[stats['negativity_ratio'] for _, stats in neg_users],
                name='Negativity',
                marker_color='red'
            ),
            row=2, col=1
        )
        
        # Highest engagement users
        top_engagement = by_engagement[:top_n]
        fig.add_trace(
            go.Bar(
                x=[user for user, _ in top_engagement],
                y=[stats['avg_engagement'] for _, stats in top_engagement],
                name='Engagement',
                marker_color='orange'
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title_text="User Sentiment & Activity Analysis",
            showlegend=False,
            height=800
        )
        
        # Update x-axis labels to be more readable
        fig.update_xaxes(tickangle=45)
        
        fig.write_html(output_path)
    
    def create_topic_sentiment_heatmap(self, posts: List["Post"], output_path: str) -> None:
        """Create heatmap showing sentiment patterns by topic and user."""
        user_stats = self.analyze_user_patterns(posts)
        
        # Create user-topic-sentiment matrix
        users = [u for u, s in user_stats.items() if s['post_count'] >= 2]
        all_topics = set()
        for stats in user_stats.values():
            all_topics.update(stats['topics'].keys())
        
        topics = sorted(list(all_topics))
        
        if not users or not topics:
            return
        
        # Build sentiment matrix
        sentiment_matrix = []
        user_labels = []
        
        for user in users[:30]:  # Limit to top 30 users for readability
            user_row = []
            stats = user_stats[user]
            
            for topic in topics:
                # Get sentiment for this user-topic combination
                topic_posts = [p for p in stats['posts'] if getattr(p, 'query', '') == topic]
                if topic_posts:
                    # Calculate average sentiment score for this topic
                    sentiment_scores = []
                    for post in topic_posts:
                        if hasattr(post, 'sentiment') and post.sentiment:
                            label = post.sentiment.get('label', 'NEUTRAL')
                            if label == 'POSITIVE':
                                sentiment_scores.append(1)
                            elif label == 'NEGATIVE':
                                sentiment_scores.append(-1)
                            else:
                                sentiment_scores.append(0)
                    
                    if sentiment_scores:
                        avg_sentiment = np.mean(sentiment_scores)
                        user_row.append(avg_sentiment)
                    else:
                        user_row.append(0)
                else:
                    user_row.append(np.nan)  # No posts for this topic
            
            sentiment_matrix.append(user_row)
            user_labels.append(f"{user} ({stats['post_count']})")
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=sentiment_matrix,
            x=topics,
            y=user_labels,
            colorscale=[
                [0, 'red'],      # Negative
                [0.5, 'white'],  # Neutral
                [1, 'green']     # Positive
            ],
            zmid=0,
            colorbar=dict(
                title="Sentiment",
                tickvals=[-1, 0, 1],
                ticktext=["Negative", "Neutral", "Positive"]
            ),
            hoverongaps=False
        ))
        
        fig.update_layout(
            title="User-Topic Sentiment Heatmap",
            xaxis_title="Topics",
            yaxis_title="Users (post count)",
            height=max(400, len(user_labels) * 25),
            xaxis=dict(tickangle=45)
        )
        
        fig.write_html(output_path)