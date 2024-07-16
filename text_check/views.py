# Create your views here.


import os

import joblib
from django.conf import settings
from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny
from rest_framework.response import Response

# Load the model and vectorizer at the start of your application
# Construct the absolute path to the model and vectorizer files
model_path = os.path.join(settings.BASE_DIR, 'text_check/ai_model.pkl')
vectorizer_path = os.path.join(settings.BASE_DIR, 'text_check/train_data_vectorizer.pkl')

# Load the model and vectorizer using the absolute path
model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)


def predict(text):
    global model, vectorizer
    print("Transforming input text...")
    features = vectorizer.transform([text])
    print("Predicting...")
    prediction = model.predict(features)
    probabilities = model.predict_proba(features)

    label = 'AI' if prediction[0] == 1 else 'Human'
    confidence = probabilities[0][prediction[0]]

    return label, confidence, probabilities


# Example usage
# text_input = "Journeying by boat offers a unique and captivating experience that intertwines adventure, tranquility, and an intimate connection with nature. Unlike other modes of travel, a boat journey encapsulates the spirit of exploration, inviting travelers to traverse vast bodies of water, navigate through serene lakes, meander along winding rivers, or venture into the open sea. The rhythmic lapping of waves against the hull, the gentle sway of the vessel, and the expansive horizon stretching endlessly before the eyes create a meditative ambiance, encouraging reflection and a deep sense of peace. From the bustling activity of a busy port, where the scent of saltwater mingles with the aroma of fresh seafood and the cacophony of seagulls fills the air, to the secluded stillness of a hidden cove, where one can anchor and enjoy the pristine beauty of unspoiled nature, every moment on a boat is a sensory delight. The opportunity to witness marine life up close, whether it be playful dolphins riding the bow wave, a school of fish darting beneath the surface, or the majestic sight of a whale breaching in the distance, adds an element of wonder and excitement to the journey. Sunrises and sunsets viewed from the deck paint the sky in a myriad of colors, offering spectacular displays that are often more vivid and dramatic than those seen from land. The camaraderie built among fellow passengers and crew, forged through shared experiences and the necessity of working together in a confined space, can lead to lasting friendships and unforgettable memories. Whether navigating through the intricate network of canals in a historic European city, sailing along the rugged coastline of a remote island, or embarking on an extended voyage across an ocean, the sense of freedom and adventure is palpable. Each destination reached by boat brings its own unique charm and challenges, from docking in a lively marina filled with yachts and sailing vessels from around the world to anchoring in a quiet bay where the only sounds are the calls of distant birds and the gentle rustling of trees. The boat itself becomes a microcosm of life, where everyday activities such as cooking, sleeping, and relaxing take on new dimensions in the confined yet cozy quarters of the cabin. The practical skills required for a successful journey, such as navigation, weather forecasting, and boat maintenance, add a layer of satisfaction and accomplishment for those who take an active role in managing the vessel. The unpredictable nature of water travel, where weather conditions and sea states can change rapidly, demands a level of respect and preparedness, reminding travelers of the power and majesty of the natural world. However, it is this very unpredictability that makes each journey unique and fosters a spirit of resilience and adaptability. The historical significance of boat travel, harking back to the days of ancient explorers and traders who charted unknown waters and connected distant lands, adds a layer of romanticism and historical context to modern-day voyages. The ability to explore remote and otherwise inaccessible places, where the land remains untouched by the footprints of mass tourism, offers a rare and precious opportunity to experience the world in its most pristine form. Whether it is a leisurely day sail, a week-long coastal cruise, or a transoceanic expedition, the journey by boat is a timeless adventure that appeals to the human spirit’s innate desire for exploration and discovery. For many, the allure of the sea is irresistible, calling them to leave behind the hustle and bustle of daily life and embark on a journey where the only limits are the horizon and the open water. The boat becomes a sanctuary, a place of retreat and reflection, where the worries of the world are left ashore and the focus shifts to the simple yet profound pleasures of life on the water. Each journey, regardless of its length or destination, becomes a story in itself, filled with moments of awe, challenge, and joy, and leaves an indelible mark on the hearts and minds of those who undertake it."
# text_input = "In the same semester, a unit on Appropedia was incorporated into an upper-level German course, German Conversation and Composition, which had 10 students enrolled. This course was offered at three Pennsylvania State universities in the classroom and via interactive television: Clarion University, Slippery Rock University, and Edinboro University. Interactive television allows for real-time student–teacher and student–student interaction and conversation, in contrast to a web-based course. This course is generally taken by students who have completed the four-semester language sequence and who are planning to major or minor in German. It was already planned to be content-based: It was organized around a series of topics pertaining to German culture before and during World War II, which led into related class discussions and writing assignments. The Appropedia unit, however, allowed the students to take a more active role in learning by creating content that can be read and used by the global community of German speakers."
# text_input = "The results indicate that the model performs as expected when the labels are randomized—it achieves an accuracy around 50%, which is what you'd expect from a model that is effectively guessing."

# text_input = """
# studies have been proven that people are starting to not drive cars as much americans are buying fewer cars and getting fewer licenses there are several advantages to not driving cars everywhere beneficial implications for carbon emissions the environment and improves safetyvaubans streets in germany are completely car free people who do not drive cars any more and ride bikes walks or rides the tram are said to be happier this way a mother of two walks and rides her bikes everywhere with her kids and does not feel as tense and stressed as she did when she was drivingalso a huge advantage is it drastically reduces greenhouse gas emissions from tailpipes diesel fuel was banned in france since tax policies favored diesel over gasoline delivery companies complained of lost revenue but exceptions were made plugin cars hybrids and cars carrying three or more passengersin paris there were days of near record pollution so they enforced a partial driving ban to clear the air of the city april 2013 the number of miles driven per person was nearly 9 percent below average and equal to the country in january 1995 the rate of car ownership per household and per person came down two to three years before the downturn michael siva stated new york has a new bike sharing program and the skyrocketing bridges and tunnel tolls reflect people to want to not drive cars anymorea study last year found that driving by young people decreased 23 percent between 2001 and 2009 just shows you that the beneficial implications environment and easier way of saving money does make people want to start biking walking using a tram etc those examples i gave you are a few advantages of limiting car use
#
# """
# label, confidence = predict(text_input)
# print(f"Prediction: {label} (Confidence: {confidence:.4f})")


@api_view(["GET"])
@permission_classes([AllowAny])
def check_text(request):
    text_params = request.query_params.get("text", "")
    label, confidence, probabilities = predict(text_params)

    return Response(
        {
            "label": label,
            "confidence": confidence,
            "probabilities": probabilities,
        },
        status=status.HTTP_200_OK,
    )
