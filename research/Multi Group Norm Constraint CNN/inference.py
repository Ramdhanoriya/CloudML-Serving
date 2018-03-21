__author__ = 'KKishore'

import tensorflow as tf

from tensorflow.contrib import predictor

tf.logging.set_verbosity(tf.logging.INFO)

base_dir = 'serving/1521601545'

prediction_fn = predictor.from_saved_model(export_dir=base_dir, signature_def_key='serving_default')

output = prediction_fn({
    'review': [
        'a very accurate depiction of small time mob life filmed in new jersey the story characters and script are believable '
        'but the acting drops the ball still it\'s worth watching especially for the strong images some still with me even though '
        'i first viewed this 25 years ago a young hood steps up and starts doing bigger things tries to but these things keep going '
        'wrong leading the local boss to suspect that his end is being skimmed off not a good place to be if you enjoy your health or life '
        'this is the film that introduced joe pesce to martin scorsese also present is that perennial screen wise guy frank vincent strong on '
        'characterizations and visuals sound muddled and much of the acting is amateurish but a great story',

        'the love letter is one of those movies that could have been really clever but they wasted it focusing on a letter '
        'wreaking havoc in a small town the movie has an all star cast with nothing to do tom selleck and alice drummond had so recently co '
        'starred in the super hilarious in out also about an upset in a small town in which they were both great but here they look as though '
        'they\'re getting drug all over the place i ca n\'t tell what the people behind the camera are trying to do here if anything but they '
        'sure did n\'t accomplish anything how tragic that a potential laugh riot got so sorrowfully wasted',

        'there is not much to add to what others have already commented the movie fails hard where it should n\'t it has no depth in the planning of the heist and the characters are so unbelievable one thing that got me thinking was that although the rest of the gang is trying hard to remove the pins from the doors of an armored truck because there is supposed to be no other way of opening it the guy inside the truck with great ease manages to remove the floor of the truck which happens to have a hole in it so he can get out and then get back in without being noticed by anyone because no one else could think that he could get out from there or even better that they could have gotten into from there promising but not quite there',

        'why was this movie made are producers so easily fooled by sadists that they\'ll give them money to create torture methods such as this so called film i love a bad movie as much as the next masochist but cave dwellers is pushing it it\'s seriously physically painful to watch the plot is something about a dude name ator a buffed up numbnuts whom i will refer to as private snowball for the rest of this review who has to fight invisible warriors and rescue a princess in order to beat the bad guy who needs to find a better hair stylist i might have gotten the plot wrong since it\'s been a while since i watched this excrement but really do you care that much oh yeah private snowball also has a mute asian sidekick who has n\'t who\'s not funny anyway private snowball fights invisible people visits some caves all in the name of a good king so personality free he makes al gore look like jim carrey then private snowball builds a hang glider yes i\'m serious and gets the girl yippie kee yay it\'s cheap unintentionally silly and mind numbingly dull why am i not surprised that the director ended up making porn bottom line avoid ator will steal a part of your life and you will have no funny so bad they\'re good catchphrases to take with you from the experience bad ator ! bad ! aak ! gags',

        'it sucks suck bad bad',

        'the jaws rip off is the trashiest of the all the italian\'genres\' and director joe d\'amato is second only to the great jess franco in the trash film production stakes put the two together and what do you get a gigantic piece of trash of course unfortunately it\'s not trash in the good sense of the word either as deep blood delivers more in boredom than it does in hilarity to the film\'s credit it does actually attempt something bordering on a plot but to take said credit away from the film the plot is rubbish it has something to do with a group of friends taking of an oath of friendship and then some indian curse that manifests itself into a shark or at least i think that\'s what was going on anyway the majority of the film is padded out with boring dialogue and\'drama\' and the shark itself which lets not forget is the only thing we really want to see finds itself in merely a cameo role or not even that since most the shark is actually stock footage ! despite being a trash genre there are actually a lot of fun jaws rip offs but with this one joe d\'amato makes it clear that he could n\'t be bothered to even try and the result is what must be the worst italian shark movie of all time avoid this dross'
    ]
})

print(output['class'])
