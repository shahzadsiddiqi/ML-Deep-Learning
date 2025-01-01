import os
import openai
import streamlit as st
import json
from dotenv import load_dotenv

load_dotenv()  # Load all the environment variables from .env file

# Set up the API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# function to get a response
def get_openai_response(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-4",  # Use "gpt-3.5-turbo" or "gpt-4" for chat models
        messages=[{"role": "system", "content": "You are a helpful assistant."},
                  {"role": "user", "content": prompt}],
        max_tokens=50
    )
    return response['choices'][0]['message']['content'].strip()

@st.cache_data
def fetch_questions(text_content, quiz_level, num_questions):
    # JSON template for response format
    RESPONSE_JSON = {
        "mcqs": [
            {
                "mcq": "Sample question 1",
                "options": {
                    "a": "Option A",
                    "b": "Option B",
                    "c": "Option C",
                    "d": "Option D"
                },
                "correct": "a"
            }
        ]
    }

    # Prompt template for OpenAI
    PROMPT_TEMPLATE = f"""
    Text: {text_content}
    You are an expert in generating  unique MCQ-type quizzes based on the provided content.
    Create a quiz of {num_questions} multiple choice questions with a {quiz_level} difficulty level.
    Ensure the response format matches the example RESPONSE_JSON provided below:
    
    {json.dumps(RESPONSE_JSON, indent=2)}
    """

    # Make API request
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # or "gpt-4" for a more advanced model
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": PROMPT_TEMPLATE}
            ],
            temperature=0.5,
            max_tokens=1000,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )

        # Extract response JSON
        extracted_response = response['choices'][0]['message']['content']
        questions = json.loads(extracted_response).get("mcqs", [])
        return questions

    except Exception as e:
        st.error(f"Error generating quiz questions: {e}")
        return []

def main():
    # Centered Title
    st.markdown("<h1 style='text-align: center;'>Quiz Vendor</h1>", unsafe_allow_html=True)

    # Subtitle below the title
    st.markdown("<h3 style='text-align: center;'> Where you test your scalability </h3>", unsafe_allow_html=True)


    # Text input for user to paste content
    text_content = st.text_area("Paste the text content here:")

    # Dropdown for selecting quiz level
    quiz_level = st.selectbox("Select quiz level:", ["Easy", "Medium", "Hard"]).lower()

    # Numeric input for number of questions
    num_questions = st.number_input("Enter the number of questions to generate:", min_value=1, max_value=15, value=5)

    # Generate Quiz button
    if st.button("Generate Quiz"):
        # Fetch questions from OpenAI API
        questions = fetch_questions(text_content, quiz_level, num_questions)

        # Save the questions in session state to persist across reruns
        if questions:
            st.session_state.questions = questions
            st.session_state.compare_answers_shown = False
            st.session_state.marks = 0
            st.session_state.selected_options = []
            st.session_state.correct_answers = []

    # Display the quiz if questions are available
    if 'questions' in st.session_state:
        selected_options = []
        correct_answers = []

        for question in st.session_state.questions:
            options = list(question["options"].values())

            # Display radio button with no pre-selected option
            selected_option = st.radio(
                question["mcq"], 
                options,
                key=question["mcq"],
                # Ensures that  no option is pre-selected
                index=None  
            )
            selected_options.append(selected_option)
            correct_answers.append(question["options"][question["correct"]])

        # Store selected options and correct answers in session state to persist after submit
        st.session_state.selected_options = selected_options
        st.session_state.correct_answers = correct_answers

        # Submit button to check answers
        if st.button("Submit"):
            # Calculate score and store it in session state
            marks = 0
            for i, selected_option in enumerate(st.session_state.selected_options):
                if selected_option == st.session_state.correct_answers[i]:
                    marks += 1
            st.session_state.marks = marks
            st.session_state.total_questions = len(st.session_state.questions)

            # Display the score immediately
            st.header("Quiz Result:")
            st.subheader(f"You scored {st.session_state.marks} out of {st.session_state.total_questions}")

            # Motivational message based on score
            if st.session_state.marks == st.session_state.total_questions:
                st.write("Excellent! You got all the answers correct! 🎉")
            elif st.session_state.marks > st.session_state.total_questions // 2:
                st.write("Good job! You did well! 👍")
            else:
                st.write("Keep trying! You can do it next time! 💪")

            # Show "Compare Answers" button
            st.session_state.compare_answers_shown = True

        # Compare Answers button logic
        if 'compare_answers_shown' in st.session_state and st.session_state.compare_answers_shown:
            if st.button("Compare Answers"):
                # Display comparison of selected answers with correct answers
                st.write("### Answer Comparison:")
                for i, question in enumerate(st.session_state.questions):
                    selected_option = st.session_state.selected_options[i]
                    correct_option = st.session_state.correct_answers[i]
                    st.write(f"**Q{i + 1}:** {question['mcq']}")
                    st.write(f"- Your answer: {selected_option}")
                    st.write(f"- Correct answer: {correct_option}")
                    if selected_option == correct_option:
                        st.write("✅ Correct")
                    else:
                        st.write("❌ Incorrect")


if __name__ == "__main__":
    main()

    # its a runner to remember streamlit run d:/quizapp/quizapp.py 