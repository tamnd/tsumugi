package langid

// trainingText holds a compact representative paragraph per Latin-script language,
// the corpus the trigram profiles are built from at construction. It is deliberately
// small: the Cavnar-Trenkle measure needs only enough text to rank the few hundred
// most frequent trigrams, and the function words and frequent affixes that dominate
// that ranking are present in any few sentences of a language. Embedding the text
// rather than shipping pretrained profiles keeps the package self-contained and lets
// the profiles rebuild deterministically, and a deployment with a domain corpus can
// extend or replace these without touching the detector.
//
// The sentences are ordinary prose chosen to carry each language's characteristic
// short words (the, and, of for English; le, la, des for French; der, die, und for
// German) and accented forms, which are exactly the signal the trigram ranks encode.
var trainingText = map[string]string{
	English: `The quick search engine returns the best results for the query that the user typed.
		When the index is large the system must rank documents quickly and return the most relevant ones.
		Search is the problem of finding the right information among a great many documents,
		and the answer is a ranking of the pages that best match the words of the question.
		People use a search engine every day to find news, products, and answers to their questions,
		and they expect the results to be fast, accurate, and easy to read on any device.`,

	Spanish: `El motor de búsqueda devuelve los mejores resultados para la consulta que el usuario escribió.
		Cuando el índice es grande el sistema debe ordenar los documentos rápidamente y devolver los más relevantes.
		La búsqueda es el problema de encontrar la información correcta entre muchos documentos,
		y la respuesta es una clasificación de las páginas que mejor coinciden con las palabras de la pregunta.
		Las personas usan un motor de búsqueda todos los días para encontrar noticias, productos y respuestas.`,

	French: `Le moteur de recherche renvoie les meilleurs résultats pour la requête que l'utilisateur a saisie.
		Lorsque l'index est grand le système doit classer les documents rapidement et renvoyer les plus pertinents.
		La recherche est le problème de trouver la bonne information parmi de nombreux documents,
		et la réponse est un classement des pages qui correspondent le mieux aux mots de la question.
		Les gens utilisent un moteur de recherche chaque jour pour trouver des nouvelles et des produits.`,

	German: `Die Suchmaschine liefert die besten Ergebnisse für die Anfrage, die der Benutzer eingegeben hat.
		Wenn der Index groß ist, muss das System die Dokumente schnell ordnen und die relevantesten zurückgeben.
		Die Suche ist das Problem, die richtigen Informationen unter sehr vielen Dokumenten zu finden,
		und die Antwort ist eine Rangfolge der Seiten, die am besten zu den Wörtern der Frage passen.
		Die Menschen benutzen jeden Tag eine Suchmaschine, um Nachrichten und Produkte zu finden.`,

	Italian: `Il motore di ricerca restituisce i migliori risultati per la query che l'utente ha digitato.
		Quando l'indice è grande il sistema deve ordinare i documenti rapidamente e restituire i più pertinenti.
		La ricerca è il problema di trovare le informazioni giuste tra moltissimi documenti,
		e la risposta è una classifica delle pagine che corrispondono meglio alle parole della domanda.
		Le persone usano un motore di ricerca ogni giorno per trovare notizie, prodotti e risposte.`,

	Portuguese: `O motor de busca retorna os melhores resultados para a consulta que o usuário digitou.
		Quando o índice é grande o sistema deve ordenar os documentos rapidamente e retornar os mais relevantes.
		A busca é o problema de encontrar a informação certa entre muitos documentos,
		e a resposta é uma classificação das páginas que melhor correspondem às palavras da pergunta.
		As pessoas usam um motor de busca todos os dias para encontrar notícias, produtos e respostas.`,

	Dutch: `De zoekmachine geeft de beste resultaten terug voor de zoekopdracht die de gebruiker heeft getypt.
		Wanneer de index groot is moet het systeem de documenten snel ordenen en de meest relevante teruggeven.
		Zoeken is het probleem van het vinden van de juiste informatie tussen heel veel documenten,
		en het antwoord is een rangschikking van de pagina's die het best overeenkomen met de woorden van de vraag.
		Mensen gebruiken elke dag een zoekmachine om nieuws, producten en antwoorden te vinden.`,
}
