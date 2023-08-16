import re
def prevent_trail_off(response):
    list_matches = re.findall("[1-9]\.\d*|\d+\.",response)
    if len(list_matches) > 1: #checks if it's listing off things 
        r = response.split(list_matches[-1])
        return r[0]
    else:
        return re.sub('\.[^.]*$','.',response)
        
a = prevent_trail_off("Sure, I'd be happy to explain how to properly saute food!Sauteing is a cooking technique that involves quickly cooking ingredients in a small amount of oil or fat over high heat. This method helps to preserve the nutrients in the food, especially heat-sensitive vitamins and minerals like vitamin C and B vitamins. To properly saute food, follow these steps:1. Choose the right pan: A stainless steel or cast iron pan is ideal for sauteing, as they retain heat well and distribute it evenly.2. Add oil or fat: Use a small amount of oil or fat with a high smoke point, such as olive oil, avocado oil, or ghee. This will help to prevent the oil from burning or smoking during cooking.3. Heat the pan: Preheat the pan over medium-high heat for about 2-3 minutes before adding the ingredients. This will ensure that the pan is hot enough to quickly cook the food.4. I like to move it move it")
print(a)