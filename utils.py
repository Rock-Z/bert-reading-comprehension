def print_example(data, i):
    """Print the fields of a passage entry in race for humans to preview
    
    Inputs:
        data (List<Example>): loaded dataset to index from
        i (Int): index of desired passage/question/options/answers entry from `data`
    """   
    
    print("id: \n", data[i].id)
    print("passage: \n",data[i].passage)
    print("questions: \n",data[i].question)
    print("options: \n",data[i].options)
    print("answers: \n",data[i].answers)