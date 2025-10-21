The task contains Study phase and Test phase.
In Study phase, subjetcs will view 125 pictures. A question will pop up at random “Did the previous picture contain a person?” Press “1” for YES；Press “0” for NO
In Test phase, subjects will view 120 pictures. For each picture you will answer two questions in sequence: “Did you see this picture during the study phase?” Press “1” for YES (old); Press “0” for NO (new) 
“How confident are you about your answer?”
Left arrow “←” = not confident
Down arrow “↓” = somewhat confident
Right arrow “→” = very confident

To run practice, run the following. $subid can go from 1-50. Let's start from subject 2.

cd ./Scripts/Presentation
# python practice_block.py $subID
python practice_block.py 2
This will give a practice of 20 study images and 20 test images with feedback. There won't be any feedback in the actual experiment. To run the actual study,

cd ./Scripts/Presentation
# python run_exp_block $subID $blockID
python run_exp_block.py 2 1 
$blockID can go from 1 to 4. Each block lasts about 16 mins.
