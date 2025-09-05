from predict import predict_intense,get_response

print("chatbot is ready type quite to exit")

def chat_loop():
    while True:
        user_input=input("you: ")
        
        if user_input.lower()=='quit':
            print("goodbye .....")
            break
        
        predicted_tag=predict_intense(user_input)
        response=get_response(predicted_tag)
        
        print(f"chatbot: {response}")
        
chat_loop()