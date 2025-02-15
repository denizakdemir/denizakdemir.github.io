import os
import graphviz

def create_tabular_transformer_diagram(output_file='assets/img/tabular_transformers.png'):
    """
    Creates and saves an improved, vertically organized diagram illustrating the architecture of a Tabular Transformer.
    
    The diagram is arranged in a top-to-bottom (TB) layout and features two data pipelines (categorical and numerical)
    along with merging and classification nodes. The final image is saved in the specified output folder.
    
    Parameters:
    -----------
    output_file : str
        The complete path (including file name and extension) for the output PNG file.
    """
    # Ensure the target directory exists; if not, create it.
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Initialize a new directed graph with a top-to-bottom layout and overall styling.
    dot = graphviz.Digraph('TabularTransformer', format='png')
    dot.attr(
        rankdir='TB',           # Arrange nodes top-to-bottom
        fontsize='18',
        labelloc='t',
        label='Tabular Transformer Architecture',
        style='filled',
        fillcolor='#f7f7f7'
    )
    
    # Global node styling.
    dot.attr('node',
             shape='box',
             style='rounded,filled',
             fontname='Helvetica',
             fontsize='12',
             fillcolor='#ffffff',
             color='#333333',
             margin='0.2,0.1')
    
    # Global edge styling.
    dot.attr('edge',
             fontname='Helvetica',
             fontsize='10',
             color='#666666',
             arrowsize='0.8')
    
    # Define subgraph for the categorical data pipeline.
    with dot.subgraph(name='cluster_cat') as cat:
        cat.attr(
            label='Categorical Data Pipeline',
            style='filled',
            fillcolor='#e3f2fd',
            fontname='Helvetica',
            fontsize='12'
        )
        cat.node('cat_input', 'Input: Categorical Data\n(each feature as token)')
        cat.node('embeddings', 'Embedding Layers\n(Learnable embeddings)')
        cat.node('positional', 'Add Positional Embeddings\n(Feature identification)')
        cat.node('transformer', 'Transformer Encoder\n(Self-Attention & FFN)')
        cat.node('flatten', 'Flatten Transformer Output')
        
        # Define edges within the categorical pipeline.
        cat.edge('cat_input', 'embeddings', label='Tokenization', color='#1e88e5')
        cat.edge('embeddings', 'positional', label='Embedded tokens', color='#1e88e5')
        cat.edge('positional', 'transformer', label='Enhanced embeddings', color='#1e88e5')
        cat.edge('transformer', 'flatten', label='Contextualized features', color='#1e88e5')
    
    # Define subgraph for the numerical data pipeline.
    with dot.subgraph(name='cluster_num') as num:
        num.attr(
            label='Numerical Data Pipeline',
            style='filled',
            fillcolor='#e8f5e9',
            fontname='Helvetica',
            fontsize='12'
        )
        num.node('num_input', 'Input: Numeric Data\n(Standardized features)')
    
    # Define nodes for merging and classification.
    dot.node('concat', 'Concatenate Features\n(Flattened embeddings + Numeric)', fillcolor='#fff3e0')
    dot.node('mlp', 'MLP Classifier\n(Fully Connected Layers)', fillcolor='#ffecb3')
    dot.node('output', 'Output: Prediction\n(Binary Classification)', fillcolor='#dcedc8')
    
    # Connect pipelines to the merge node.
    dot.edge('flatten', 'concat', label='Flattened Output', color='#fb8c00')
    dot.edge('num_input', 'concat', label='Numeric Features', color='#43a047')
    
    # Connect the merged features to the classifier and then to the output.
    dot.edge('concat', 'mlp', label='Combined Features', color='#fb8c00')
    dot.edge('mlp', 'output', label='Classification', color='#fb8c00')
    
    # Extract the base filename (without extension) for rendering.
    base_filename, _ = os.path.splitext(output_file)
    dot.render(filename=base_filename, cleanup=True)
    
    print(f"Improved diagram saved as {output_file}")

if __name__ == '__main__':
    create_tabular_transformer_diagram()
