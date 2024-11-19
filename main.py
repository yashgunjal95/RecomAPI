from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Load the saved model and data
with open('model.pkl', 'rb') as file:
    data, similarity_matrix = pickle.load(file)

# Recommendation function
def recommend_by_district_taluka(district, taluka, num_recommendations=5):
    # Filter data based on the provided district and taluka
    filtered_data = data[(data['District'].str.lower() == district.lower()) & 
                         (data['Taluka'].str.lower() == taluka.lower())]
    
    if filtered_data.empty:
        return {'error': 'No data found for the specified district and taluka.'}

    # Compute similarity only within the filtered data
    filtered_indices = filtered_data.index.tolist()
    filtered_similarity_matrix = similarity_matrix[filtered_indices][:, filtered_indices]

    # Recommend items based on similarity
    recommendations = []
    for i in range(min(num_recommendations, len(filtered_indices))):
        similarity_scores = list(enumerate(filtered_similarity_matrix[i]))
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        top_indices = [filtered_indices[s[0]] for s in similarity_scores[1:num_recommendations + 1]]
        recommendations.extend(data.iloc[top_indices].to_dict(orient='records'))

    return recommendations[:num_recommendations]


@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        # Get input data from the request
        req_data = request.get_json()
        district = req_data.get('district')
        taluka = req_data.get('taluka')
        num_recommendations = req_data.get('num_recommendations', 5)

        # Validate input
        if not district or not taluka:
            return jsonify({'error': 'District and Taluka are required fields.'}), 400

        # Generate recommendations
        recommendations = recommend_by_district_taluka(district, taluka, num_recommendations)
        if isinstance(recommendations, dict) and 'error' in recommendations:
            return jsonify(recommendations), 404

        return jsonify({'recommendations': recommendations})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
