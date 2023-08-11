from transformers import AutoTokenizer, T5ForConditionalGeneration
from transformers import (   
    AutoModelForSeq2SeqLM,   
    T5Tokenizer,
)
tokenizer = AutoTokenizer.from_pretrained("lmsys/fastchat-t5-3b-v1.0")
model = AutoModelForSeq2SeqLM.from_pretrained("lmsys/fastchat-t5-3b-v1.0")

def summerize_with_fastchat(passage):
    MAX_PASSAGE_LENGTH = 2040 #char
    passage = f"Human: Using this passage, awnser this question: I live in the netherlands and I can't tolerate cold temperatures. What are some good colleges? {passage[:MAX_PASSAGE_LENGTH]} Assistant: "  # Ensure the task is properly formatted
    encoded_input = tokenizer.encode(passage, return_tensors='pt')
    output = model.generate(encoded_input, max_length=512, temperature=0.7, top_p=1)
    decoded_output = tokenizer.decode(output[0, :], skip_special_tokens=True)
    decoded_output = decoded_output.replace('<pad> ','')
    decoded_output = decoded_output.replace("  "," ")
    return decoded_output

r = summerize_with_fastchat(
"""
Can I pour hot beverages in glass cups? - Quora
Can I pour hot beverages in glass cups?
12 Answers
Best
Joe DiVincenti
, Retired from 50 Years in Logistics & Supply Chain
Answered 3 years ago · Author has 697 answers and 1.5M answer views
Originally Answered: Can you pour hot beverages in glass cups?
As a wise old recruiter once explained to my board of directors….
“Ladies and gentlemen, given sufficient time and funds you can train an ostrich to climb a tree. However, if the prospectus amounts to “Tree Climber Needed” you might consider hiring a monkey”.
His was a lesson in why we needed to stop promoting our best operational people into the sales channel. It caught on so much that twenty years later I’d still hear people referring to the sales manager in, say, Walla Walla, as “Ed… another ostrich in a tree.”
It is however also a great line as to using the proper tool for the job at hand. Ye
Continue Reading
Edward Willhoft
, PhD (1966) FIFST (1987) Physical Chemistry, King’s College London (1962)
Answered 3 years ago · Author has 3.2K answers and 5.1M answer views
Originally Answered: Can you pour hot beverages in glass cups?
If the glass cup has survived previous thermal shocks and provided it has not suffered an impact that has produced a microfissure (that grows over time and use), and if the cup was well annealed during manufacture (capable of determination as residual stress under cross -polarised light) and if the cup glass is thin (to minimise thermal gradients) and has a uniform thickness, or is made of borosilicate glass (as with Pyrex) then it should survive the thermal shock due to sudden contact with boiling water.
Note the several caveats! Therefore, if you want to play safe then use a china or porcelai
Continue Reading
Jessie White
, studied Marketing at Income School
Answered 3 years ago · Author has 121 answers and 52.1K answer views
Originally Answered: Can you pour hot beverages in glass cups?
I would used just any glass cup for a hot beverage. But i do recommened you invest in a double wall glass.
They’re usually sold as coffee glasses, espresso cups, latte cups and cappucino cups.
I use double wall glasses for my coffee every morning (if im not on the go, otherwise I pour it into my thermos). They’re perfect for keeping your tea or coffee at the correct temperature too.
This way if you leave your tea or coffee for a few minutes to finish a task, when you come back yo it the drink is still warm.
4K views
View upvotes
Answer requested by
Eugenio Nitti
Dan Fabian
, Retired and living in Canada
Answered 3 years ago · Author has 267 answers and 131.6K answer views
Originally Answered: Can you pour hot beverages in glass cups?
Before you pour a hot drink to a glass cup, make sure you leave a metal spoon in it. The glass cup will not crack. It works (almost) all the time.
1.8K views
View upvotes
Richard O. Colestock
, former Skeptical scientist, manager and world traveler.
Answered 9 months ago · Author has 29.1K answers and 5.1M answer views
Originally Answered: Can you pour hot beverages in glass cups?
Yes, as long as the cup is made of borosilicate glass. This is done every day in laboratories using glass beakers and flasks made of this low expansion material.
991 views
Lynne Chamberlain
, former Midwifery Nurse Trainee at St Mary's Hospital, London (1975-1976)
Answered 1 year ago · Author has 8K answers and 1.2M answer views
Originally Answered: Can you pour hot beverages in glass cups?
No.
You need to put a metal spoon in the glass first. This absorbs the heat and prevents the glass from breaking.
1.5K views
Quora User
Answered 3 years ago
Originally Answered: Can you pour hot beverages in glass cups?
Yes you can pour hot beverages in glass cups there is no law against it. But depending on the type of glass the cup is made of it might not be a good idea.
The only glass you should really pour hot beverages or water into is borosilicate. It's made for hot and cold and is not going to shatter or crack when you pour hot or cold liquids into it.
For most other types of glass, it's going to be a mystery. It might break, it might not. You're really not going to know until you pour the liquid into it.
7.7K views
View upvotes
Quora User
Answered 3 years ago · Author has 1.7K answers and 282.2K answer views
Originally Answered: Can you pour hot beverages in glass cups?
Porcelain is much preferred.. hey, go out to the hardware store and look to see if they have any “glass” cups for sale.
don’t look in the wine glass section . there wont be any borosilicate glass there.
Pyrex makes a glass measuring cup that is approved for hot or cold liquids..
382 views
Ian Woollard
, Electronics Engineer
Updated 7 months ago · Author has 2.2K answers and 1.4M answer views
Originally Answered: Can you pour hot beverages in glass cups?
Yes, absolutely. Provided it’s borosilicate glass, even if it’s resting in a puddle of cold water, or even sitting in salted ice water when you do it, it will be absolutely fine. Borosilicate glass is low expansion glass that can take large temperature differences.
But if it’s soda-lime glass, which is commonly marketed in the United States as ‘Pyrex’ then there’s a very high chance your glass will explode.
A borosilicate glass object can take 170C (340F) of temperature difference across it. Boiling water is at 100C, and most freezers are set to -15C, so salted ice water will be at -15C. So that
Continue Reading
Curt Weinstein
, HS Physics & Mathematics, The Bronx High School of Science (1969)
Answered 10 months ago · Author has 26.6K answers and 11.1M answer views
Originally Answered: Can you pour hot beverages in glass cups?
Usually, that works. My grandmother would place a metal spoon in the cup. She thought that kept the cup from breaking when pouring boiling water into the cup. The cups didn’t break.
949 views
Janusz Weremowicz
, lives in The United States of America (1985-present)
Answered 3 years ago · Author has 351 answers and 40K answer views
Originally Answered: Can you pour hot beverages in glass cups?
Of course you can . Russians and Poles are drinking almost boiled tea in a glass cups.
300 views Answer requested by
Eugenio Nitti
Richard Harrison
, Professional Photographer
Answered 3 years ago · Author has 1.7K answers and 256K answer views
Originally Answered: Can you pour hot beverages in glass cups?
Absolutely! As long as it is borosilicate glass. This type of glass contains boron trioxide which allows for a very low coefficient of thermal expansion. In other words, it won’t crack or shatter because of a sudden temperature change. It’s why Pyrex was so useful in a kitchen until 1998 when they quit using borosilicate. High end teacups etc are still made from borosilicate glass.
7.8K views
View upvotes
Related Answer
Mehran Moalem
, UC Berkeley, Taught Engineering, some Chemistry and Physics.
Answered 1 year ago · Author has 507 answers and 3.7M answer views
Why can't I put boiling water in a glass bottle?
You can put boiling water in a glass bottle. But you have to do extra steps. Before you pour the water into the bottle hold it out in the sink and pour the boiling water on the outside of it first. Afterwards, pour a little bit of boiling water into the bottle shake a bit so it wets the wall and starts heating it slowly and finally proceed to fill it with the boiling water. The reason why you have to do it this way has to do with thermal expansion and stress fracture. Glass expands like any o...
Upgrade to Quora+ to access this answer
Access millions more answer s like this
Browse ad‑free
Support the writers you like to hear from
Start free trial
Learn more
31.1K views
View upvotes
Related Answer
Charles Wilson
, former Engineer/manager Semiconductor Process Automation
Answered 1 year ago · Author has 1.5K answers and 829.4K answer views
Why does thin cold tempered glass brake when washed in hot water?
You have the wrong attitude. With glass the question is why does the glass not break under temperature changes. The only class resilient for temperature changes quartz glass which is what the Pyrex measuring cups are made of. Pyrex has a very low thermal expansion coefficient. The small coefficient means when the inside of the glass changes temperature 400 degrees it doesn't grow or shrink and put stress across the the glass thickness.. Because glasses are relatively good insulators, the inside and the outside of the glass are different temperatures . When one tries to change dimension and the
Continue Reading
Related Answer
Bernard Ossman
, have been drinking coffee for a long time
Answered 1 year ago · Author has 974 answers and 1.2M answer views
Isn’t it advisable to drink hot beverages in a paper cup? Why is it not advisable, and what are the ill effects on health?
The immediate risk of using a paper cup for hot beverages ia the seam is glued. Higher temperatures are more likely to soften the adhesive, possibly resulting in leakage. This could cause a burn or a slipping hazard.
Paper cups, especially those made from recycled paper, have a fairly high concentration of Bis-Phenol A, which is associated with cardiovascular and endocrine system disorders. It tends to accumulate in the body, so relatively low exposure over a long period of time can result in pretty serius health issues. It is not very soluble in water, but solubility increase with elevated tem
Continue Reading
Related Answer
Alessandro Fais
, chewed physics to create things
Answered 6 years ago · Author has 760 answers and 3.3M answer views
Why does a regular glass break when a hot liquid is poured into it?
Originally Answered: Can a cup made out of glass break if I pour hot water in it?
Absolutely yes, and it has happened to me more than once (although I should know better) when I drain the boiling water of pasta on the kitchen sink with some glasses left over inside.
If you pour hot water on a glass, the point were the water hits the glass will increase its temperature rapidly and will try to expand creating thermal tensions on the structure. Rapid heating (or cooling) is called thermal shock and it is a topic of interest in both mechanical engineering and materials engineering.
Thermal shock resistance of most glasses is low because of their low flexual strength, low thermal
Continue Reading
Related Answer
Jasmine Adamson
, Glass Artist
Answered 3 years ago · Author has 3K answers and 8M answer views
When you pour boiling water into a glass jar it will crack, but if it’s made of Pyrex glass it would not crack. Why?
Originally Answered: When you pour boiling water into a glass jar it will crack but if it’s made of Pyrex glass would not crack why?
So, Pyrex… is bog standard borosilicate glass. The marketing department at Corning might disagree with me on this but the lab probably would agree, Pyrex is nothing special, it is just glass.
There are (kinda) only two types of glass, borosilicate and soda-lime (soft glass). The main difference between them is tolerance of thermal shock. Borosilicate is much more tolerant of thermal shock, but all glass is somewhat tolerant of thermal shock. Indeed, I regularly heat soft glass to over 1200 degrees without having it crack. It's all about how you do it.
However, no glass is immune to catastrophic
Continue Reading
Related Answer
Alexander L
, lifelong Californian
Answered 6 months ago · Author has 13K answers and 25M answer views
What is wrong with consuming hot beverages from a plastic cup?
For plastic cups designed to withstand high temperature liquids, absolutely nothing.
But for plastic cups that are not designed to hold high temperature liquids, some of the plastic may leach into the drink. Whether or not that’s harmful to your health is a highly debated topic— some studies suggest it’s carcinogenic, while others say there’s insufficient evidence to prove it’s harmful. But in any case, it does impart a nasty, plasticky flavour to your drink, which I’m sure you don’t want.
264 views
View upvotes
Related Answer
Beth Goldowitz
, Tea is soothing. Tea is energizing.
Answered 3 years ago · Author has 18K answers and 44.8M answer views
Can a mason jar withstand boiling hot water being poured into it to make tea?
I sterilize my mason jars by pouring boiling water in them and haven’t had a problem yet, but I do run warm water in them first, especially if it’s a cold day. Mason jars are designed to be used for hot canning, so they aren’t fragile, but I would still warm the jar before using it to make tea.
1.2K views
View upvotes
Related Answer
Quora User
, I like to cook and eat it too
Answered 2 years ago
Which type of cups are best to drink hot beverages?
Indian kulhad(earthen pots ) or Indian ceramic cups as are used by Railways. The simple test is heat it with small amount of water in microwave for 30 seconds, the pot or cup should not get hot, but water only should get hot.
126 views
View upvotes
Related Answer
Quora User
, studied at Rhode Island School of Design
Updated 2 years ago · Author has 221 answers and 133K answer views
Why did my glass cup break when I pour hot tea over it?
Glass breaks when you pour hot tea into it because it is -by nature - a brittle material and a poor conductor (of heat in this case ).
When the tea heats the glass, the glass expands. But because glass is a poor conductor, the parts not heated by the tea, like the rim and even the outside surface, stay the same size. The tension between the expanded parts and the non-expanded parts cracks the glass.
There are glasses such as Pyrex that are formulated to have a low expansion coefficient, which means that they withstand temperature changes better.
A good example of glass's ability to insulate heat
Continue Reading
Related Answer
Steve Baker
, Senior Software Engineer (2013-present)
Answered 4 years ago · Upvoted by
Quora User
, Degree level applied chemistry + 20yr experience in corrosion prevention and water treatment · Author has 25.3K answers and 127.3M answer views
What happens when you pour hot water into a cold glass?
Glass expands when heated - but it’s not a really good conductor of heat.
So what happens is that the glass that’s nearest to the heat will try to expand - but the material on the outside of the glass will not.
This causes a lot of internal stress inside the glass - which can make it shatter immediately, or perhaps build up sufficient internal stress that it shatters when you touch it.
Many times, the stress will build up but not be forceful enough to shatter the glass - and eventually the temperature throughout the thickness will even out and the stress will be relieved without damaging the glas
Continue Reading
Related Answer
Dan Granier
, Student at Bristol Grammar School (2012-present)
Answered 4 years ago
What will happen to a double layer glass cup when hot tea is poured in it?
Originally Answered: When hot water is poured inside a double-layered glass cup, what happens?
Double layered cups can be used to effectively insulate warm or cold drinks. If the two layers are separated by a vacuum and the top is covered it prevents heat loss by conduction and convection, a shiny surface can be used to reflect infra red radiation also reducing heat loss.
1.1K views
View upvotes
Related Answer
Gil Silberman
, I took the factory tour
Answered 4 years ago · Author has 8.6K answers and 22.6M answer views
Is it safe to put hot chocolate in a glass?
It depends on the glass, and how hot. I’d do it over the sink, but normally a glass can with stand hot liquid. Glasses aren’t ideal for drinking hot liquid, though, because they transmit heat too well — it may be too hot to hold, and is unpleasant to drink from.
Bodum (image below) and others make attractive double-walled glasses that do a better job than mugs keeping liquids hot, yet easy to handle — also good at keeping cold liquids cold. The downside is that they are expensive, and fragile.
Usually, when glasses shatter it’s because you suddenly pour something cold in a hot glass, not the oth
Continue Reading
Related Answer
Richard Allen
, former Class Facilitator Cal State Northridge 2002-2005
Answered 2 years ago · Author has 273 answers and 1M answer views
When pouring hot water into a glass jar, sometimes the jar cracks. Is there a way to prevent the probability of cracking from happening?
Regular commercial glass is quite a bit stronger above room temperature, plus cracking can be prevented by raising the temperature of the entire glass container evenly and slowly.
Placing the glass jars in a pot of cold water so the jars are submerged, and then heating the pot of water is how it is done. The jars are then filled with whatever they are going to be filled with that is the same temperature as the jars. After filling, the jars are cooled down to room temperature with a cloth cover over them to prevent cold air convection from creating cold spots on the jars.
If filling with hot acid
Continue Reading
Related Answer
Quora User
, Degree level applied chemistry + 20yr experience in corrosion prevention and water treatment
Answered 4 years ago · Author has 19K answers and 19.8M answer views
What if hot water is poured in thick glass?
If it was borosilicate glass ( pryrex is the brand I am familiar with but there are others) nothing would happen. Simple silicate ( soda glass) would probably shatter, but it depends on how hot the water is exactly and how quickly you poured the water, you might be able to do it many times without the glass shattering, it depends on the exact thickness and shape of the glass as well.
2.9K views
View upvotes
Related Answer
Barry Lauritzen
, former Infantryman, U.S. Army 1975, Designer/Drafter  at Honeywell E&TSC (1980-1991)
Answered 2 years ago · Author has 2.3K answers and 992.5K answer views
Can I put a glass jar on the stove to warm it?
Yes you can, no you shouldn’t.
533 views
View upvotes
Related Answer
Michael Mombourquette
, Chemistry Professor at Queen's University at Kingston (1996-present)
Answered 4 years ago · Author has 5.7K answers and 10.8M answer views
Will glass explode if I boil it?
Glass will break only if one part of it gets heated strongly but another part remains cold. The hot glass expands but the cold glass does not. That puts a lot of stress on the structure and it can break if the stress gets high enough. If you heat all the glass gently enough and evenly so that no part gets hotter than another part then no. It will not break. Ask any glass blower.
18K views
View upvotes
Related Answer
Pat Harkin
, worked at University of Leeds (1984-2018)
Answered 4 years ago · Author has 16.2K answers and 14.3M answer views
A tea cup is stuck in a glass. How do It get it out? (I've tried hot water and ice)
I would suggest you put crushed ice/water mix in the teacup and lower the glass into hot water (you’ll probably need to hold it down) so the the point at which the teacup touches it is under water. The aim is to combine the expansion of the glass and the contraction of the cup at the same time. Once the glass has warmed through, push and pull the cup -= it should come free.
234 views
View upvotes
Related Answer
Jordan Kull
, Structural Engineer, designing Commercial buildings on the daily.
Answered 4 years ago · Author has 290 answers and 1.9M answer views
Will a mason jar shatter if I pour hot water in it?
Glass, like most brittle materials, can shatter when it changes temperature abruptly. Tempered or not you can shatter a mason jar if you try. And you can have it not shatter if you are trying to keep it intact.
If the change in temperature is “too abrupt” it will shatter. If the change in temperature is gradual the jar will remain intact.
If you remove a jar from the freezer and pour boiling water in it will shatter
If you remove a hat jar from the dish washer and pour hot water in, it will not chatter.
The stress the Jar experiences is proportional to the change in temperature it is undergoing. W
Continue Reading
Related Answer
Quora User
, Analytical chemist, computer programmer, ketogenic, atheist.
Answered 6 years ago · Author has 19K answers and 19.8M answer views
Why does a thick glass tumbler sometimes crack when you pour very hot tea into it?
Originally Answered: Why does a thick glass tumbler sometimes Crack when you pour very hot tea into it?
The crack is caused by differential thermal expansion.
The part of the glass that the boiling water touches first expands due to the heat, but the heat doesn't conduct through the glass quickly so there is stress between the expanded inside of the glass  and unexpanded outside of the glass. This stress is sometimes too much and the glass cracks.
22.8K views
View upvotes
Related Answer
Quora User
, trained Chef, over 15 years experience from pubs to hotels
Answered 1 year ago · Author has 4.2K answers and 11M answer views
Can you pour boiling water into a glass?
Well, obviously yes.
As has been mentioned, pouring boiling water into glasses can be a bad idea under some circumstances.
If you have a room-temperature or below glass that isn't meant to hold hot liquids, sitting on top of a relatively cold surface which isn't a decent thermal insulator, chances are it will crack the moment you pour boiling water in.
There are plenty of glasses that are made to be suitable for holding hot liquids, and those will be fine, unless you really try (like chilling the glass and putting it on a chilled metal surface before pouring in the boiling water).
Even 'normal’ gl
Continue Reading
Related Answer
Douglas Porter
, former Teacher of Mathematics (2013-2018)
Answered 2 years ago · Author has 6.2K answers and 2M answer views
Why does cold glass crack when you put hot water on it?
The heat causes the glass to expand on the side that is being warmed, but glass conducts heat poorly so the outside of the glass isn’t being warmed and isn’t expanding. The expanding glass is therefore trying to forcibly stretch the cool glass, and glass doesn’t stretch well, so it cracks. The crack will then run right through the glass almost instantly; cracks in glass propagate very easily.
922 views
1 share
Related Answer
Daniel Lindsäth
, I'm moderately healthy
Answered 3 years ago · Author has 5.5K answers and 8.1M answer views
How come drinking 'hot' drinks from a glass, is healthier than drinking from a mug?
How come drinking 'hot' drinks from a glass, is healthier than drinking from a mug?
What makes you think that it is? This sounds like one of those crazy health myths that keeps getting spread around but have absolutely nothing to back them up.
86 views
View upvotes
Answer requested by
Kieran Byrne
Related Answer
Bradford White
, Engineer Specializing in  HVAC and Energy (1977-present)
Answered 2 years ago · Author has 3.2K answers and 2.7M answer views
When I poured hot water into a glass jar, the jar exploded, sending glass shards everywhere. How did this happen and how do I prevent it?
Do not pour boiling water into a glass jar is one way to prevent it.
If you must, however, one way is to immerse the jar in warm water and heat it gradually, suspended from contact with the bottom of the pan. (To leave that in contact can also cause cracking.)
Once the water is at the boiling point and the glass is at the same approximate temperature, hot water may be poured slowly into the jars and allowed to cool together.
What happens is uneven thermal expansion. Glass is of course brittle and expands rapidly when heated. Your jar is on the counter at room temperature. If you poured hot water
Continue Reading
Related Answer
Grady Beckett
, lives in England
Answered 4 years ago · Author has 337 answers and 344.5K answer views
Will a ceramic mug for tea with a hot beverage break if put in the freezer?
Because it contains a hot beverage, the whole system goes from hot to cold together. I wouldn't think that it would break.
Disclaimer, this is just my opinion, not scientifically rigourous fact and should be treated as such. If you have nice mugs, perhaps don't use them here.
7.6K views
View upvotes
Answer requested by
Leonardo Lombardi
Related Answer
Joe Shuffield
, engineer
Answered 4 years ago · Author has 4.4K answers and 2.3M answer views
What will break if hot water is poured, thin or thick glass?
breaking is caused because heat does not dissipate fast enough and the concentrated heat causes a fracture in the glass. A thick piece of glass will aborb and dissipate heat more readily than a thin piece of glass.
214 views
Related Answer
Alex Randomkat
Answered 5 years ago · Author has 74 answers and 290.4K answer views
What happens when you put cold water into a hot glass?
Originally Answered: what happens when you put cold water into a hot glass?
What happens to the glass depends on the temperature difference. Let's assume we have a glass plate about a centimeter thick. When the glass is heated and rapidly cooled by a liquid (or the other way around), the outside of the glass cools more rapidly than if in air. This is because molecules in a liquid are much more packed together than in air, so heat can exchange faster between glass and a liquid than glass and air because with the liquid the glass will make contact with more molecules and be able to transfer more energy to the liquid compared to the gas. Furthermore the liquid may evapor
Continue Reading
Related Answer
Alexander L
Answered 2 years ago · Author has 13K answers and 25M answer views
Do glass mugs keep drinks hot?
Nothing but heat keeps drinks hot.
You can certainly use different materials to make hot drinks cool down slower. Glass is better than metal, but plastic is a much better insulator than both. You can also get vacuum-insulated mugs, which can insulate hot drinks for up to 6–7 hours.
409 views
View upvotes
Related Answer
Tyler Thompson
, Missionary  at World Mission Continuum (2014-present)
Updated 3 years ago · Author has 58 answers and 371.2K answer views
Why is hot water best for making cups of tea?
To properly extract the flavour from the tea leaves you need to heat the water. It takes much longer to infuse the tea into warm or cold water….the colder the temperature, the longer it takes.
The correct temperature of water is also necessary: Too hot or cold of a temperature can result in bad tasting tea. For example, White teas should be steeped in water that is at a temperature of 70 degrees Celsius. If the water is too hot the tea will go bitter.
318 views
Related Answer
Patrick Lee
, lives in Kathmandu, Nepal
Answered 2 years ago · Author has 5.2K answers and 1M answer views
Do plastic cups keep drinks hot?
No. Plastic cups do not keep drinks hot as the heat will dissipate rather quickly due to the thin walls of the cup
21 views
Related Answer
Vedank Singhania
, studied at National Public School, Koramangala
Answered 3 years ago · Author has 61 answers and 259.4K answer views
Does a dining table glass break when you place hot utensils on it?
Glass dining table is capable of withstanding a thermal shock of 700Â°C (or 1290Â°F). Thus, placing a hot pot on the glass should not lead to the glass breakage. Still, I would only do such a thing - placing a hot utensil on a cold cooktop, as an ultimate measure, and not something to be done often. I have not had any incident of glass breakage due to a thermal shock. From my experience, using a large pot/pan on a small induction element should not lead to the glass damage.
It is my hope that this will answer your questions.
4.5K views
View upvotes
Related Answer
Karen J Gray
, former Courier, Secretary,Advertising Director, Sales  (1976-1990)
Answered 4 years ago · Author has 29.5K answers and 14.4M answer views
What happens when hot water is put in a thin glass and a doubled layer glass cup?
If the water is hot enough, chances are good that the thin glass will break, due to thermal shock, though thin glass is less likely to break than thicker glass, simply because it takes that much longer for the heat to travel through thicker glass, making the pressure even worse.
The inside of the glass heats up fast, but, glass isn’t a very good conductor of heat. So the heat travels slowly through it. The outside of the glass is colder. than the how hot inside is.
The hot inner layer expands quite a bit, but the cooler outside does not, which puts a lot of pressure on the exterior of the glass.
Continue Reading
Related Answer
Lori McConnell
, M.S Materials Science and Engineering, University of California, Berkeley (1980)
Answered 3 years ago · Author has 52 answers and 721.3K answer views
How many minutes can we put hot water into food in plastic cups?
So I’m personally kind of a fanatic about this—I stopped eating hot food out of plastic containers when I first learned about plastics chemistry. I would never microwave food in plastic unless I was desperate, and I even store food in glass containers in the frig when possible. The whole thing with BPA in the last few years confirmed some of these suspicions.
But the truth may well be that there are now highly -designed plastics that are stable with heat and would not contaminate food. The problem is that mass-produced food containers are generally made of the cheapest possible plastics. Check
Continue Reading
""")
print(r)