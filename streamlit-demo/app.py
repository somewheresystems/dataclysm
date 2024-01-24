# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from datasets import load_dataset, Dataset
from sklearn.cluster import KMeans
import plotly.graph_objects as go
import time, random, datetime
import logging
from sklearn.cluster import HDBSCAN


BACKGROUND_COLOR = 'black'
COLOR = 'white'

def set_page_container_style(
        max_width: int = 10000, max_width_100_percent: bool = False,
        padding_top: int = 1, padding_right: int = 10, padding_left: int = 1, padding_bottom: int = 10,
        color: str = COLOR, background_color: str = BACKGROUND_COLOR,
    ):
        if max_width_100_percent:
            max_width_str = f'max-width: 100%;'
        else:
            max_width_str = f'max-width: {max_width}px;'
        st.markdown(
            f'''
            <style>
                .reportview-container .css-1lcbmhc .css-1outpf7 {{
                    padding-top: 35px;
                }}
                .reportview-container .main .block-container {{
                    {max_width_str}
                    padding-top: {padding_top}rem;
                    padding-right: {padding_right}rem;
                    padding-left: {padding_left}rem;
                    padding-bottom: {padding_bottom}rem;
                }}
                .reportview-container .main {{
                    color: {color};
                    background-color: {background_color};
                }}
            </style>
            ''',
            unsafe_allow_html=True,
        )

# Additional libraries for querying
from FlagEmbedding import FlagModel

# Global variables and dataset loading
global dataset_name
st.set_page_config(layout="wide")

dataset_name = "somewheresystems/dataclysm-arxiv"

set_page_container_style(
        max_width = 1600, max_width_100_percent = True,
        padding_top = 0, padding_right = 10, padding_left = 5, padding_bottom = 10
)
st.session_state.dataclysm_arxiv = load_dataset(dataset_name, split="train")
total_samples = len(st.session_state.dataclysm_arxiv)

logging.basicConfig(filename='app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s', level=logging.INFO)
# Load the dataset once at the start
# Initialize the model for querying
model = FlagModel('BAAI/bge-small-en-v1.5', query_instruction_for_retrieval="Represent this sentence for searching relevant passages:", use_fp16=True)


def load_data(num_samples):
    start_time = time.time() 
    dataset_name = 'somewheresystems/dataclysm-arxiv'
    # Load the dataset
    logging.info(f'Loading dataset...')
    dataset = load_dataset(dataset_name)
    total_samples = len(dataset['train'])

    logging.info('Converting to pandas dataframe...')
    # Convert the dataset to a pandas DataFrame
    df = dataset['train'].to_pandas()

    # Adjust num_samples if it's more than the total number of samples
    num_samples = min(num_samples, total_samples)
    st.sidebar.text(f'Number of samples: {num_samples} ({num_samples / total_samples:.2%} of total)')

    # Randomly sample the dataframe
    df = df.sample(n=num_samples)

    # Assuming 'embeddings' column contains the embeddings
    embeddings = df['title_embedding'].tolist()
    print("embeddings length: " + str(len(embeddings)))

    # Convert list of lists to numpy array
    embeddings = np.array(embeddings, dtype=object)
    end_time = time.time()  # End timing
    st.sidebar.text(f'Data loading completed in {end_time - start_time:.3f} seconds')
    return df, embeddings

def perform_tsne(embeddings):
    start_time = time.time() 
    logging.info('Performing t-SNE...')

    n_samples = len(embeddings)
    perplexity = min(30, n_samples - 1) if n_samples > 1 else 1

    # Check if all embeddings have the same length
    if len(set([len(embed) for embed in embeddings])) > 1:
        raise ValueError("All embeddings should have the same length")

    # Dimensionality Reduction with t-SNE
    tsne = TSNE(n_components=3, perplexity=perplexity, n_iter=300)

    # Create a placeholder for progress bar
    progress_text = st.empty()
    progress_text.text("t-SNE in progress...")

    tsne_results = tsne.fit_transform(np.vstack(embeddings.tolist()))

    # Update progress bar to indicate completion
    progress_text.text(f"t-SNE completed at {datetime.datetime.now()}. Processed {n_samples} samples with perplexity {perplexity}.")
    end_time = time.time()  # End timing
    st.sidebar.text(f't-SNE completed in {end_time - start_time:.3f} seconds')
    return tsne_results


def perform_clustering(df, tsne_results):
    start_time = time.time() 
    # Perform DBSCAN clustering
    logging.info('Performing HDBSCAN clustering...')
    # Step 3: Visualization with Plotly
    # Normalize the t-SNE results between 0 and 1
    df['tsne-3d-one'] = (tsne_results[:,0] - tsne_results[:,0].min()) / (tsne_results[:,0].max() - tsne_results[:,0].min())
    df['tsne-3d-two'] = (tsne_results[:,1] - tsne_results[:,1].min()) / (tsne_results[:,1].max() - tsne_results[:,1].min())
    df['tsne-3d-three'] = (tsne_results[:,2] - tsne_results[:,2].min()) / (tsne_results[:,2].max() - tsne_results[:,2].min())

    # Perform DBSCAN clustering
    hdbscan = HDBSCAN(min_cluster_size=10, min_samples=50)
    cluster_labels = hdbscan.fit_predict(df[['tsne-3d-one', 'tsne-3d-two', 'tsne-3d-three']])
    df['cluster'] = cluster_labels
    end_time = time.time()  # End timing
    st.sidebar.text(f'HDBSCAN clustering completed in {end_time - start_time:.3f} seconds')
    return df

def update_camera_position(fig, df, df_query, result_id, K=10):
    # Focus the camera on the closest result
    top_K_ids = df_query.sort_values(by='proximity', ascending=True).head(K)['id'].tolist()
    top_K_proximity = df_query['proximity'].tolist()
    top_results = df[df['id'].isin(top_K_ids)]
    camera_focus = dict(
        eye=dict(x=top_results.iloc[0]['tsne-3d-one']*0.1, y=top_results.iloc[0]['tsne-3d-two']*0.1, z=top_results.iloc[0]['tsne-3d-three']*0.1)
    )
    # Normalize the proximity values to range between 1 and 10
    normalized_proximity = [10 - (10 * (prox - min(top_K_proximity)) / (max(top_K_proximity) - min(top_K_proximity))) for prox in top_K_proximity]
    # Create a dictionary mapping id to normalized proximity
    id_to_proximity = dict(zip(top_K_ids, normalized_proximity))
    # Set marker sizes based on proximity for top K ids, all other points stay the same -- 500% zoom
    marker_sizes = [5 * id_to_proximity[id] if id in top_K_ids else 1 for id in df['id']]
    # Store the original colors in a separate column
    df['color'] = df['cluster']

    fig = go.Figure(data=[go.Scatter3d(
        x=df['tsne-3d-one'],
        y=df['tsne-3d-two'],
        z=df['tsne-3d-three'],
        mode='markers',
        marker=dict(size=marker_sizes, color=df['color'], colorscale='Viridis', opacity=0.8, line_width=0),
        hovertext=df['hovertext'],
        hoverinfo='text',
    )])
    # Set grid opacity to 10%
    fig.update_layout(scene = dict(xaxis = dict(gridcolor='rgba(128, 128, 128, 0.1)', color='rgba(128, 128, 128, 0.1)'),
                                    yaxis = dict(gridcolor='rgba(128, 128, 128, 0.1)', color='rgba(128, 128, 128, 0.1)'),
                                    zaxis = dict(gridcolor='rgba(128, 128, 128, 0.1)', color='rgba(128, 128, 128, 0.1)')))

    # Add lines stemming from the first point to all other points in the top K
    for i in range(1, K):  # there are K-1 lines from the first point to the other K-1 points
        fig.add_trace(go.Scatter3d(
            x=[top_results.iloc[0]['tsne-3d-one'], top_results.iloc[i]['tsne-3d-one']],
            y=[top_results.iloc[0]['tsne-3d-two'], top_results.iloc[i]['tsne-3d-two']],
            z=[top_results.iloc[0]['tsne-3d-three'], top_results.iloc[i]['tsne-3d-three']],
            mode='lines',
            line=dict(color='white',width=0.3),  # Set line opacity to 50%
            showlegend=True,
            name="centroid" if i == -1 else top_results.iloc[i]['id'],  # Set the legend to "Top Result" for the first entry, and to the title of the article for the rest
            hovertext=f'Title: Top K Results\nID: {top_K_ids[i]}, Proximity: {round(top_K_proximity[i], 4)}',
            hoverinfo='text',
        ))
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                scene_camera=camera_focus)
    return fig

def main():
    # Custom CSS
    custom_css = """
    <style>
        /* Define the font */
        @font-face {
            font-family: 'F';
            src: url('https://fonts.googleapis.com/css2?family=Martian+Mono&display=swap') format('truetype');
        }
        /* Apply the font to all elements */
        * {
            font-family: 'F', sans-serif !important;
            color: #F8F8F8; /* Set the font color to F8F8F8 */
        }
        /* Add your CSS styles here */
        .stPlotlyChart {
            width: 100%;
            height: 100%;
        /* Other styles... */
        }
        h1 {
            text-align: center;
        }
        h2,h3,h4 {
            text-align: justify;
            font-size: 8px;
        }
        st-emotion-cache-1wmy9hl {
            font-size: 8px;
        }
        body {
            color: #fff;
            background-color: #202020;
        }

        .stSlider .css-1cpxqw2 {
            background: #202020;
            color: #fd5137;
        }
        .stSlider .text {
            background: #202020;
            color: #fd5137;
        }
        .stButton > button {
            background-color: #202020;
            width: 60%;
            margin-left: auto;
            margin-right: auto;
            display: block;
            padding: 10px 24px;
            font-size: 16px;
            font-weight: bold;
            border: 1px solid #f8f8f8;
        }
        .stButton > button:hover {
            color: #Fd5137
            border: 1px solid #fd5137;
        }
        .stButton > button:active {
            color: #F8F8F8;
            border: 1px solid #fd5137;
            background-color: #fd5137;
        }
        .reportview-container .main .block-container {
            padding: 0;
            background-color: #202020;
            width: 100%; /* Make the plotly graph take up full width */
        }
        .sidebar .sidebar-content {
            background-image: linear-gradient(#202020,#202020);
            color: white;
            size: 0.2em; /* Make the text in the sidebar smaller */
            padding: 0;
        }
        .reportview-container .main .block-container {
            background-color: #000000;
        }
        .stText {
            padding: 0;
        }
        /* Set the main background color to #202020 */
        .appview-container {
            background-color: #000000;
            padding: 0;
        }
        .stVerticalBlockBorderWrapper{
            padding: 0;
            margin-left: 0px;
        }
        .st-emotion-cache-1cypcdb {
            background-color: #202020;
            background-image: none;
            color: #000000;
            padding: 0;
        }
        .stPlotlyChart {
            background-color: #000000;
            background-image: none;
            color: #000000;
            padding: 0;
        }
        .reportview-container .css-1lcbmhc .css-1outpf7 {
            padding-top: 35px;
        }
        .reportview-container .main .block-container {
            max-width: 100%;
            padding-top: 0rem;
            padding-right: 0rem;
            padding-left: 0rem;
            padding-bottom: 10rem;
        }
        .reportview-container .main {
            color: white;
            background-color: black;
        }
        .st-emotion-cache-1avcm0n {
            color: black;
            background-color: black;
        }
        .st-emotion-cache-z5fcl4 {
            padding-left: 0.1rem;
            padding-right: 0.1rem;
        }
        .st-emotion-cache-z5fcl4 {
            width: 100%;
            padding: 3rem 1rem 1rem;
            min-width: auto;
            max-width: initial;
        }
        .st-emotion-cache-uf99v8 {
            display: flex;
            flex-direction: column;
            width: 100%;
            overflow: hidden;
            -webkit-box-align: center;
            align-items: center;
        }

    </style>
    """

    # Inject custom CSS with markdown
    st.markdown(custom_css, unsafe_allow_html=True)
    st.sidebar.title('arXiv Spatial Search Engine')
    st.sidebar.markdown(
        '<a href="http://dataclysm.xyz" target="_blank" style="display: flex; justify-content: center; padding: 10px;">dataclysm.xyz <img src="https://www.somewhere.systems/S2-white-logo.png" style="width: 8px; height: 8px;"></a>', 
        unsafe_allow_html=True
    )
    # Create a placeholder for the chart
    chart_placeholder = st.empty()
    
    # Check if data needs to be loaded
    if 'data_loaded' not in st.session_state or not st.session_state.data_loaded:
        # User input for number of samples
        num_samples = st.sidebar.slider('Select number of samples', 1000, int(round(total_samples/10)), 1000)
        if 'fig' not in st.session_state:
            with open('prayers.txt', 'r') as file:
                lines = file.readlines()
                random_line = random.choice(lines).strip()
            st.session_state.fig = go.Figure(data=[go.Scatter3d(x=[], y=[], z=[], mode='markers')])
            st.session_state.fig.add_annotation(
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper",
                text=random_line,
                showarrow=False,
                font=dict(
                    size=16,
                    color="black"
                ),
                align="center",
                ax=0,
                ay=0,
                bordercolor="black",
                borderwidth=2,
                borderpad=4,
                bgcolor="white",
                opacity=0.8
            )
            # Set grid opacity to 10%
            st.session_state.fig.update_layout(scene = dict(xaxis = dict(gridcolor='rgba(128, 128, 128, 0.1)', color='rgba(128, 128, 128, 0.1)'),
                                           yaxis = dict(gridcolor='rgba(128, 128, 128, 0.1)', color='rgba(128, 128, 128, 0.1)'),
                                           zaxis = dict(gridcolor='rgba(128, 128, 128, 0.1)', color='rgba(128, 128, 128, 0.1)')))

            st.session_state.fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                height=888,
                margin=dict(l=0, r=0, b=0, t=0),
                scene_camera=dict(eye=dict(x=0.1, y=0.1, z=0.1))
            )
        chart_placeholder.plotly_chart(st.session_state.fig, use_container_width=True)
        if st.sidebar.button('Initialize'):
            st.sidebar.text('Initializing data pipeline...')

            # Define a function to reshape the embeddings and add FAISS index if it doesn't exist
            def reshape_and_add_faiss_index(dataset, column_name):
                
                # Ensure the shape of the embedding is (1000, 384) and not (1000, 1, 384)
                # As each row in title_embedding is shaped like this: [[-0.08477783203125, -0.009719848632812, ...]]
                # We need to flatten it to [-0.08477783203125, -0.009719848632812, ...]
                print(f"Flattening {column_name} and adding FAISS index...")
                # Flatten the embeddings
                dataset[column_name] = dataset[column_name].apply(lambda x: np.array(x).flatten())
                # Add the FAISS index
                dataset = Dataset.from_pandas(dataset).add_faiss_index(column=column_name)
                print(f"FAISS index for {column_name} added.")
                
                return dataset
            
            # Load data and perform t-SNE and clustering
            df, embeddings = load_data(num_samples)

            # Combine embeddings and df back into one df
            # Convert embeddings to list of lists before assigning to df
            embeddings_list = [embedding.flatten().tolist() for embedding in embeddings]
            df['title_embedding'] = embeddings_list
            # Print the first few rows of the dataframe to check
            print(df.head())
            # Add FAISS indices for 'title_embedding' 
            st.session_state.dataclysm_title_indexed = reshape_and_add_faiss_index(df, 'title_embedding')
            tsne_results = perform_tsne(embeddings)
            df = perform_clustering(df, tsne_results)
            # Store results in session state
            st.session_state.df = df
            st.session_state.tsne_results = tsne_results
            st.session_state.data_loaded = True
        
            # Create custom hover text
            df['hovertext'] = df.apply(
                lambda row: f"<b>Title:</b> {row['title']}<br><b>arXiv ID:</b> {row['id']}<br><b>Key:</b> {row.name}", axis=1
            )
            st.sidebar.text("Datasets loaded, titles indexed.")

            # Create the plot
            fig = go.Figure(data=[go.Scatter3d(
                x=df['tsne-3d-one'],
                y=df['tsne-3d-two'],
                z=df['tsne-3d-three'],
                mode='markers',
                hovertext=df['hovertext'],
                hoverinfo='text',
                marker=dict(
                    size=1,
                    color=df['cluster'],
                    colorscale='Jet',
                    opacity=0.75
                )
            )])
            # Set grid opacity to 10%
            fig.update_layout(scene = dict(xaxis = dict(gridcolor='rgba(128, 128, 128, 0.1)', color='rgba(128, 128, 128, 0.1)'),
                                           yaxis = dict(gridcolor='rgba(128, 128, 128, 0.1)', color='rgba(128, 128, 128, 0.1)'),
                                           zaxis = dict(gridcolor='rgba(128, 128, 128, 0.1)', color='rgba(128, 128, 128, 0.1)')))

            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                height=800,
                margin=dict(l=0, r=0, b=0, t=0),
                scene_camera=dict(eye=dict(x=0.1, y=0.1, z=0.1))
            )
            st.session_state.fig = fig

    # Display the plot if data is loaded
    if 'data_loaded' in st.session_state and st.session_state.data_loaded:
        chart_placeholder.plotly_chart(st.session_state.fig, use_container_width=True)


    # Sidebar for detailed view
    if 'df' in st.session_state:
        # Sidebar for querying
        with st.sidebar:
            st.sidebar.markdown("# Detailed View")
            selected_index = st.sidebar.selectbox("Select Key", st.session_state.df.id)

            # Display metadata for the selected article
            selected_row = st.session_state.df[st.session_state.df['id'] == selected_index].iloc[0]
            st.markdown(f"### Title\n{selected_row['title']}", unsafe_allow_html=True)
            st.markdown(f"### Abstract\n{selected_row['abstract']}", unsafe_allow_html=True)
            st.markdown(f"[Read the full paper](https://arxiv.org/abs/{selected_row['id']})", unsafe_allow_html=True)
            st.markdown(f"[Download PDF](https://arxiv.org/pdf/{selected_row['id']})", unsafe_allow_html=True)

            st.sidebar.markdown("### Find Similar in Latent Space")
            query = st.text_input("", value=selected_row['title'])
            top_k = st.slider("top k", 1, 100, 10)
            if st.button("Search"):
                # Define the model
                print("Initializing model...")
                model = FlagModel('BAAI/bge-small-en-v1.5', 
                                query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
                                use_fp16=True)
                print("Model initialized.")
                
                query_embedding = model.encode([query])
                query_embedding = np.array(query_embedding).reshape(1, -1).astype('float32')
                # Retrieve examples by title similarity (or abstract, depending on your preference)
                scores_title, retrieved_examples_title = st.session_state.dataclysm_title_indexed.get_nearest_examples('title_embedding', query_embedding, k=top_k)
                df_query = pd.DataFrame(retrieved_examples_title)
                df_query['proximity'] = scores_title
                df_query = df_query.sort_values(by='proximity', ascending=True)
                # Limit similarity score to 3 decimal points
                df_query['proximity'] = df_query['proximity'].round(3)
                # Fix the <a href link> to display properly
                df_query['URL'] = df_query['id'].apply(lambda x: f'<a href="https://arxiv.org/abs/{x}" target="_blank">Link</a>')
                st.sidebar.markdown(df_query[['title', 'proximity', 'id']].to_html(escape=False), unsafe_allow_html=True)
                # Get the ID of the top search result
                top_result_id = df_query.iloc[0]['id']

                # Update the camera position and appearance of points
                updated_fig = update_camera_position(st.session_state.fig, st.session_state.df, df_query, top_result_id,top_k)

                # Update the figure in the session state and redraw the plot
                st.session_state.fig = updated_fig

                # Update the chart using the placeholder
                chart_placeholder.plotly_chart(st.session_state.fig, use_container_width=True)

   

if __name__ == "__main__":
    main()