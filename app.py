"""
MLB "Deserve to Win" Simulator - Streamlit App
Home page with project overview and methodology.
"""

import streamlit as st

# Page configuration
st.set_page_config(
    page_title="MLB Deserve-to-Win Simulator",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for cleaner styling
st.markdown("""
<style>
    /* Main title styling */
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A5F;
        margin-bottom: 0.5rem;
    }
    
    /* Subtitle styling */
    .subtitle {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    
    /* Card styling for feature boxes */
    .feature-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #E81828;
    }
    
    /* Link styling */
    .social-link {
        display: inline-block;
        padding: 0.5rem 1rem;
        margin: 0.25rem;
        background-color: #1E3A5F;
        color: white !important;
        border-radius: 5px;
        text-decoration: none;
    }
    
    /* Footer styling */
    .footer {
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #e0e0e0;
        color: #888;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


def main():
    # Header
    st.markdown('<p class="main-title">⚾ MLB Deserve-to-Win Simulator</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="subtitle">Who <em>should</em> have won? Batted ball simulation using exit velocity, launch angle, and spray angle.</p>',
        unsafe_allow_html=True
    )
    
    # Quick links row
    col1, col2, col3 = st.columns(3)
    with col1:
        st.link_button("🐦 Twitter @mlb_simulator", "https://x.com/mlb_simulator", use_container_width=True)
    with col2:
        st.link_button("🦋 Bluesky", "https://bsky.app/profile/mlb-simulator.bsky.social", use_container_width=True)
    with col3:
        st.link_button("📊 View Game Simulations", "Game_Simulations", use_container_width=True)
    
    st.divider()
    
    # Main content in two columns
    left_col, right_col = st.columns([3, 2])
    
    with left_col:
        st.header("How It Works")
        
        st.markdown("""
        Every MLB game has moments where the outcome could have gone differently. 
        A line drive caught by a diving outfielder, a fly ball that just clears the fence, 
        a hard-hit grounder that finds a hole.
        
        This simulator asks: **if we replayed every batted ball 10,000 times, who would win?**
        """)
        
        st.subheader("The Model")
        st.markdown("""
        For each batted ball in a game, I use a **Gradient Boosting Classifier** trained on 
        MLB Statcast data to predict the probability of each outcome:
        
        - **Features**: Exit velocity, launch angle, spray angle (adjusted for batter handedness), and ballpark
        - **Outcomes**: Out, Single, Double, Triple, Home Run
        - **Accuracy**: 82% on held-out test data
        
        The model accounts for **ballpark effects** — a fly ball that's a home run at Yankee Stadium 
        might be a routine out at Oracle Park's cavernous right field.
        """)
        
        st.subheader("The Simulation")
        st.markdown("""
        For each game:
        1. Get all batted ball events from the MLB Stats API
        2. For each batted ball, predict outcome probabilities using the model
        3. Run **10,000 simulations** where each batted ball is randomly sampled based on its probabilities
        4. Calculate how often each team wins across all simulations
        
        The result is a **"deserve-to-win" percentage** — the probability each team would win 
        if the game were replayed thousands of times with the same quality of contact.
        """)
        
        st.subheader("New in 2025: Spray Angle")
        st.markdown("""
        The latest version incorporates **spray angle** — where the ball was hit on the field. 
        A pull-side line drive has different outcome probabilities than an opposite-field liner 
        with the same exit velocity and launch angle.
        
        Spray angle is adjusted for **batter handedness** so "pull side" is consistent for 
        both lefties and righties, improving model accuracy from 77% to 82%.
        """)
    
    with right_col:
        st.header("Sample Visualizations")
        
        # Note about images
        st.info("👈 Browse actual game simulations in the **Game Simulations** tab")
        
        st.markdown("**Each game generates 4 charts:**")
        
        with st.expander("🎯 Spray Chart", expanded=True):
            st.markdown("""
            Stadium-specific spray chart showing where each batted ball landed, 
            colored by the model's predicted outcome. Shows the actual ballpark 
            dimensions and how they affect hit probabilities.
            """)
        
        with st.expander("📊 Run Distribution"):
            st.markdown("""
            Histogram showing how many runs each team scored across all 10,000 simulations. 
            Reveals whether the actual score was typical or an outlier.
            """)
        
        with st.expander("📈 Expected Bases by Player"):
            st.markdown("""
            Table ranking each batter by their expected bases (probability-weighted outcomes). 
            Shows who made the best contact regardless of actual results.
            """)
        
        with st.expander("👥 Player Contributions"):
            st.markdown("""
            Horizontal bar chart showing each player's contribution to their team's 
            expected run production, split by batted balls vs. walks.
            """)
        
        st.divider()
        
        st.header("Read More")
        st.markdown("""
        📝 [**Who "Deserved" to Win? Building an MLB Game Outcome Simulator**](https://medium.com/@dmgrifka_64770/who-deserved-to-win-building-an-mlb-game-outcome-simulator-b4a8d4bca2a9)  
        *Original methodology and motivation*
        
        📝 [**Applying Bayesian Hierarchical Methods to MLB Season Win Probabilities with PyStan**](https://medium.com/@dmgrifka_64770/applying-bayesian-hierarchical-methods-to-mlb-season-win-probabilties-with-pystan-468572abb932)  
        *Using simulation results to estimate true team strength*
        """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        '<p class="footer">Built by <a href="https://dgrifka.github.io">Derek Grifka</a> · '
        'Data from MLB Stats API · Model trained on Statcast data</p>',
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
