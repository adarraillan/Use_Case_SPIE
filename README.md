# README

##Architecture projet
<ul>
    <li>Prediction
        <ul>
            <li>dataset
            <ul>
                <li>DataLoader.py : </li>
                <li>data</li>
                <li>data_preprocess</li>
            </ul>
            </li>
            <li>models
                contients la classe Model pour train et tout et des sous-classes des diff models
            </li>
            <li>saved_models
                contient les models sauvegardés (format h5)
            </li>
            <li>preprocess
                <ul><li>prepocess.py : transformer les csv en quelques choses de mangeable par le modele (apres le dataloader)</li></ul>
            </li>
            <li>result
                contient les résultats</li>
            <li>_init.py</li>
        </ul>
    </li>
</ul>

##Format données
type(0 appart, 1 maison), nb_inhabitant, superficy, date, weather(null?) -> vecteur de conso (taille 48)
