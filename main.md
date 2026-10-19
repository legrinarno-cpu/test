# Application Development — aide-mémoire du midterm

**React + JavaScript · Cours et sujets vérifiés le 19 octobre 2026.**

**Pour trouver vite :** `Ctrl+F` puis `balises`, `state`, `props`, `formulaire`, `map`, `useEffect`, `useRef`, `Context`, `RGB`, `password`, `paires` ou `morpion`.

**Méthode :** lire la consigne → choisir les données qui changent → créer les composants → brancher les événements → calculer le résultat → tester les limites.

## 1. Démarrer un composant

Un **composant** est une fonction qui retourne l’interface en JSX. Son nom commence par une majuscule. Exemple complet pour `src/App.js` :

```jsx
import { useState } from "react";
import "./App.css";

export default function App() {
  const [count, setCount] = useState(0);
  return (
    <main>
      <h1>Compteur</h1>
      <p>Valeur : {count}</p>
      <button onClick={() => setCount(c => c + 1)}>+1</button>
    </main>
  );
}
```

**Dans les autres exemples :** importer les hooks utilisés depuis `"react"`. Les extraits sont indépendants ; placer les déclarations de state dans un composant et le JSX dans son `return`.

| Export dans le fichier | Import dans un autre fichier |
|---|---|
| `export default App;` | `import App from "./App";` |
| `export function Field() { ... }` | `import { Field } from "./Field";` |

Un export par défaut maximum par fichier. Plusieurs exports nommés sont possibles.

## 2. Balises : laquelle choisir ?

| Besoin | Balise | Exemple |
|---|---|---|
| Boîte générique pour regrouper/organiser | `div` | `<div className="controls">...</div>` |
| Partie thématique, généralement avec un titre | `section` | `<section><h2>Résultat</h2>...</section>` |
| Contenu principal de la page | `main` | `<main>...</main>` |
| Titre principal / sous-titre | `h1` / `h2` | `<h2>Résultat</h2>` |
| Paragraphe | `p` | `<p>Tu as gagné !</p>` |
| Petit morceau de texte dans une ligne | `span` | `<span>Score : 2</span>` |
| Action au clic | `button` | `<button onClick={reset}>Reset</button>` |
| Saisie / nom du champ | `input` / `label` | `<label>Nom <input /></label>` |
| Champs à soumettre | `form` | `<form onSubmit={submit}>...</form>` |
| Liste / élément | `ul` ou `ol` / `li` | `<ul><li>React</li></ul>` |
| Image | `img` | `<img src={photo} alt="Portrait" />` |
| Lien de navigation | `a` | `<a href="/about">À propos</a>` |

**`div` ou `section` ?** Pour simplement regrouper, `div` convient. Pour une partie identifiable comme « Profil » ou « Résultats », utiliser `section`. Deux `div` peuvent afficher un titre et un texte, mais `h2` et `p` décrivent leur rôle. Le CSS gère l’apparence.

**Règles JSX :**

- Fermer les balises : `<div>...</div>` ; `<input />` et `<img />` se ferment directement.
- Un parent commun dans `return` : `<div>...</div>` ou `<>...</>` (**fragment**, sans boîte ajoutée).
- Écrire `className`, `htmlFor`, `onClick`, `onChange` avec cette casse.
- `{expression}` affiche du JavaScript : `<p>{count + 1}</p>`. Les guillemets donnent du texte fixe.
- Un `p` contient du texte ; placer les `div`/`section` autour des paragraphes.
- Commentaire dans le JSX : `{/* commentaire */}`. Dans le JavaScript : `// commentaire`.

## 3. JavaScript : l’essentiel

```js
const name = "Lina";               // pas de réaffectation
let total = 0;                     // réaffectation possible
const add = (a, b) => a + b;        // fonction fléchée
const level = total > 10 ? "Haut" : "Bas"; // condition ternaire
const student = { name: "Lina", age: 20 };
const { age } = student;            // destructuration
const updated = { ...student, age: 21 }; // copie puis remplacement
```

- `===` : égalité sans conversion ; `!==` : différence ; `&&` : ET ; `||` : OU ; `!` : NON.
- `if (condition) { ... } else { ... }` ; `switch` pour plusieurs cas ; `for`/`while` pour répéter.
- Un input fournit une **chaîne** : `"2" + "3"` → `"23"` ; `Number("2") + Number("3")` → `5`.
- Vérifier le vide avant conversion : `Number("")` vaut `0`. `Number.isFinite(n)` vérifie un nombre fini.
- `parseInt("12.7", 10)` → `12` ; `parseFloat("12.7")` → `12.7`.
- Texte : `text.length`, `text.trim()`, `text.toUpperCase()` ; tableau : `items.length`.
- `const` permet de modifier un objet ; **React exige quand même une copie pour mettre à jour son state**.
- `let`/`const` : portée de bloc ; `var` : portée de fonction.

## 4. Tableaux : choisir la bonne méthode

| Méthode | Utilité | Exemple |
|---|---|---|
| `map` | Transformer / produire une liste JSX | `items.map(x => x.name)` |
| `filter` | Conserver certains éléments | `items.filter(x => x.id !== id)` |
| `find` | Premier élément correspondant | `items.find(x => x.id === id)` |
| `includes` | Présence d’une valeur simple | `ids.includes(id)` |
| `some` / `every` | Au moins un / tous | `board.every(cell => cell !== null)` |
| `reduce` | Cumuler une valeur | `nums.reduce((sum, n) => sum + n, 0)` |
| `forEach` | Faire une action pour chaque élément | Ne retourne pas de liste JSX. |

`map` appelle sa fonction pour chaque élément et retourne un nouveau tableau. Avec `{ ... }` dans une fonction fléchée, écrire `return` :

```jsx
items.map(item => <p key={item.id}>{item.name}</p>)
items.map(item => { return <p key={item.id}>{item.name}</p>; })
```

**`key` :** identifiant unique parmi les éléments frères, stable entre les rendus. Éviter une valeur aléatoire et l’index si la liste peut changer d’ordre.

## 5. State et événements

**State = mémoire du composant.** Le setter demande un nouveau rendu.

```jsx
const [count, setCount] = useState(0);
const [items, setItems] = useState([]);
const [form, setForm] = useState({ name: "" });

// Exemples à appeler dans un gestionnaire, pas pendant le rendu :
setCount(previous => previous + 1); // dépend de l’ancien state
setItems(previous => [...previous, newItem]); // ajouter
setItems(previous => previous.filter(x => x.id !== id)); // supprimer
setForm(previous => ({ ...previous, name: "Noé" })); // changer un champ
```

**Ne jamais modifier directement le state** avec `items.push(...)`, `form.name = ...` ou `board[i] = ...`. Copier d’abord. `sort`/`reverse` modifient aussi le tableau : utiliser `[...items].sort(...)`. La copie avec `...` est superficielle : copier aussi les objets imbriqués modifiés.

**Après `setCount(...)`, `count` garde sa valeur du rendu courant dans le gestionnaire.** Pour deux incréments : appeler deux fois `setCount(c => c + 1)`.

| Écriture | Comportement |
|---|---|
| `onClick={handleClick}` | Appelle la fonction au clic. |
| `onClick={() => choose(id)}` | Appelle avec un argument au clic. |
| `onClick={choose(id)}` | Appelle pendant le rendu : erreur fréquente. |
| `onChange={e => setText(e.target.value)}` | Actualise une saisie. |

**Résultat dérivé :** calculer `const complete = selected.length === 4` ; éviter un second state qui pourrait se désynchroniser. L’updater `previous => ...` doit rester pur : pas d’`alert`, de requête ou d’autre setter à l’intérieur.

## 6. Props et callbacks — exemple RGB

**Props = données reçues du parent.** L’enfant les lit. Pour demander une modification, il appelle une fonction reçue en prop : un **callback**.

Exemple complet à mettre dans un fichier, avec `import { useState } from "react";` :

```jsx
function SingleColorPicker({ color, value, onChange }) {
  return <label>{color}
    <input type="number" min="0" max="255" value={value}
      onChange={e => {
        const n = Number(e.target.value);
        if (Number.isFinite(n))
          onChange(Math.max(0, Math.min(255, n)));
      }} />
  </label>;
}

export default function RGBColorPicker() {
  const [rValue, setRValue] = useState(0);
  const [gValue, setGValue] = useState(0);
  const [bValue, setBValue] = useState(0);
  return <section>
    <h2>Choisir une couleur</h2>
    <SingleColorPicker color="r" value={rValue} onChange={setRValue} />
    <SingleColorPicker color="g" value={gValue} onChange={setGValue} />
    <SingleColorPicker color="b" value={bValue} onChange={setBValue} />
    <div style={{ width: 120, height: 120,
      backgroundColor: `rgb(${rValue}, ${gValue}, ${bValue})` }} />
  </section>;
}
```

L’input vide revient ici à `0`. `min`/`max` ne remplacent pas le contrôle logique. `color`, `value`, `onChange` et les trois états correspondent à l’annale RGB.

**`children`** = contenu écrit entre les balises du composant :

```jsx
function Panel({ children }) { return <section>{children}</section>; }
// Dans un return : <Panel><p>Bonjour</p></Panel>
```

## 7. Formulaire contrôlé

**Contrôlé = la valeur affichée vient du state, et `onChange` met ce state à jour.**

```jsx
function Registration() {
  const [form, setForm] = useState({ name: "", accepted: false });
  function change(e) {
    const { name, type, value, checked } = e.target;
    setForm(previous => ({ ...previous,
      [name]: type === "checkbox" ? checked : value }));
  }
  function submit(e) {
    e.preventDefault(); // évite le rechargement de la page
    console.log(form);
  }
  return <form onSubmit={submit}>
    <label>Nom <input name="name" value={form.name} onChange={change} /></label>
    <label><input type="checkbox" name="accepted"
      checked={form.accepted} onChange={change} /> J’accepte</label>
    <button type="submit">Envoyer</button>
  </form>;
}
```

| Contrôle | Patron JSX dans un composant |
|---|---|
| Texte / nombre | `<input value={text} onChange={e => setText(e.target.value)} />` |
| Checkbox | `<input type="checkbox" checked={ok} onChange={e => setOk(e.target.checked)} />` |
| Radio | `<input type="radio" name="course" value="React" checked={course === "React"} onChange={e => setCourse(e.target.value)} />` |
| Select | `<select value={course} onChange={e => setCourse(e.target.value)}><option value="React">React</option></select>` |
| Texte long | `<textarea value={text} onChange={e => setText(e.target.value)} />` |

Dans un formulaire, donner `type="button"` aux boutons qui ne doivent pas soumettre. `preventDefault()` annule l’action par défaut ; `stopPropagation()` empêche l’événement de remonter aux parents.

## 8. Conditions, listes et petits patrons

```jsx
// JSX à placer dans le return :
<>
  {won ? <p>Gagné !</p> : <p>Continue</p>}
  {complete && <p>Terminé</p>}
  {items.map(item => <button key={item.id}
    onClick={() => choose(item.id)}>{item.name}</button>)}
</>
```

Avec `&&`, utiliser un booléen : `items.length > 0 && ...` évite d’afficher `0`.

```js
setVisible(v => !v); // basculer vrai/faux
setIndex(i => (i + 1) % images.length); // carrousel suivant
setIndex(i => (i - 1 + images.length) % images.length); // précédent
const dice = Math.floor(Math.random() * 6) + 1; // entier de 1 à 6
```

Le carrousel suppose une liste d’images non vide. Pour une liste imbriquée, chaque `map` doit retourner ses éléments avec leurs `key`.

## 9. Images, CSS et vocabulaire

```jsx
import photo from "./photo.jpg"; // image située dans src
import "./App.css";
// JSX à placer dans le return :
<>
  <img src={photo} alt="Portrait" className="portrait" />
  <div style={{ backgroundColor: "red", width: 120, height: 120 }} />
</>
```

CSS externe : `.portrait { width: 120px; border-radius: 8px; }`. En ligne : un objet JS, propriétés en camelCase et nombres généralement en pixels. Pour aligner : `.controls { display: flex; gap: 12px; }`.

Image située dans `public/images/photo.jpg` : `src="/images/photo.jpg"`, sans `public/` dans l’URL.

| Terme | À retenir |
|---|---|
| HTML / CSS / JavaScript | Structure / apparence / comportement. |
| React / JSX | Bibliothèque d’interface / syntaxe pour décrire les éléments dans le JS. |
| DOM / Virtual DOM | Arbre réel de la page / représentation utilisée par React pour comparer les rendus. |
| Déclaratif | Décrire l’écran à partir du state ; React actualise le DOM. |
| Node / npm | Exécute le JS hors navigateur / gère paquets et scripts. |
| Bootstrap / Router / React Native | Styles prêts à l’emploi / navigation entre vues / interfaces mobiles. |

## 10. useEffect, useRef et Context

**Hooks :** les appeler en haut du composant, avant les retours conditionnels, jamais dans un `if`, une boucle ou un gestionnaire. Les cours travaillent avec les composants fonctions ; les classes sont explicitement exclues par Lecture 1.

### useEffect : synchroniser avec l’extérieur

Importer `useEffect`, puis dans le composant :

```jsx
useEffect(() => {
  document.title = `Compteur : ${count}`;
}, [count]);
```

- Sans tableau : après chaque rendu validé ; `[]` : au montage ; `[count]` : au montage puis quand `count` change.
- Inclure toutes les valeurs réactives utilisées par l’effet. En développement, StrictMode peut rejouer montage/nettoyage.
- Nettoyer timers/abonnements : `return () => clearInterval(timerId);` depuis l’effet.
- Une valeur calculable depuis le state se calcule directement, sans effet ni state supplémentaire.

### useRef : référence sans nouveau rendu

Importer `useRef`. Exemple de saisie **non contrôlée** :

```jsx
function QuickForm() {
  const inputRef = useRef(null);
  return <form onSubmit={e => {
    e.preventDefault();
    console.log(inputRef.current.value);
  }}>
    <input ref={inputRef} />
    <button type="submit">Lire</button>
  </form>;
}
```

Le DOM garde la saisie. `ref.current` permet de la lire ; modifier une ref ne relance pas le rendu. Utiliser le state pour une valeur qui doit actualiser l’écran.

### Context : partager des données entre descendants

Importer `createContext`, `useContext` et `useState` :

```jsx
const StudentContext = createContext(null);
function StudentProvider({ children }) {
  const [student, setStudent] = useState({ name: "Lina" });
  return <StudentContext.Provider value={{ student, setStudent }}>
    {children}
  </StudentContext.Provider>;
}
function Summary() {
  const { student } = useContext(StudentContext);
  return <p>{student.name}</p>;
}
// Dans le return du parent :
// <StudentProvider><Summary /></StudentProvider>
```

Le Provider doit entourer les consommateurs. Ceux-ci lisent les mêmes données avec `useContext`. Pour modifier `student`, utiliser une copie via `setStudent`.

## 11. JSON et API

**JSON = texte de données**, pas encore un objet JavaScript. Guillemets doubles, sans fonctions ni commentaires.

```js
const student = JSON.parse('{"name":"Lina"}'); // texte → objet
const text = JSON.stringify(student);           // objet → texte

async function loadItems() {
  const response = await fetch("/api/items"); // chemin d’exemple
  if (!response.ok) throw new Error("HTTP " + response.status);
  return await response.json();
}
loadItems().then(data => console.log(data))
  .catch(error => console.error(error.message));
```

`fetch` et `response.json()` sont asynchrones. `fetch` ne rejette pas automatiquement les erreurs HTTP. Dans React, lancer la requête dans un événement ou un effet approprié, pas pendant le rendu. Adapter l’URL à une vraie API.

## 12. Annales : règles et logique à retrouver vite

### Password : longueur uniquement

Faible **< 8**, moyen **8 à 12 inclus**, fort **> 12**. Tester **7, 8, 12, 13**. Ne pas ajouter des règles majuscules/symboles non demandées.

```jsx
const [password, setPassword] = useState("");
const level = password.length < 8 ? "Faible"
  : password.length <= 12 ? "Moyen" : "Fort";
const colors = { Faible: "red", Moyen: "orange", Fort: "green" };
// Dans un parent commun du return :
// <input type="password" value={password}
//   onChange={e => setPassword(e.target.value)} />
// <p style={{ color: colors[level] }}>{level}</p>
```

### Sélection et paires d’images

Sujet publié : **8 options, maximum 4 choix, résultat après les 4 choix**. Deux paires → victoire ; une paire → **2 points de jeu** ; zéro → défaite. Ces points ne sont pas le barème de l’examen.

```jsx
const [selected, setSelected] = useState([]); // ids de cartes
function choose(id) {
  setSelected(previous => {
    if (previous.length >= 4 || previous.includes(id)) return previous;
    return [...previous, id];
  });
}
const complete = selected.length === 4;
```

**ID de carte ≠ identité d’image.** Deux cartes différentes peuvent montrer la même image. Compter les paires sur les identités d’images sélectionnées :

```js
function countPairs(values) {
  const counts = new Map();
  for (const value of values)
    counts.set(value, (counts.get(value) ?? 0) + 1);
  return [...counts.values()].reduce((sum, n) => sum + Math.floor(n / 2), 0);
}
// ["a", "a", "b", "c"] → 1 ; ["a", "a", "b", "b"] → 2
```

Bloquer le cinquième choix et le double clic sur une carte ; afficher le résultat seulement quand `complete`. La distribution exacte des huit images, un reset et un mélange ne sont pas explicitement imposés par ce texte.

### Calculatrice

Séparer **saisie texte**, **total numérique**, **opérateur**. Boutons **ou** clavier selon l’annale. Convertir avant `+`, refuser vide/non-nombre et division par zéro. Préférer une logique explicite à `eval`.

```js
function calculate(left, right, op) {
  if (String(left).trim() === "" || String(right).trim() === "")
    throw new Error("Saisie vide");
  const a = Number(left), b = Number(right);
  if (!Number.isFinite(a) || !Number.isFinite(b)) throw new Error("Nombre invalide");
  switch (op) {
    case "+": return a + b;
    case "-": return a - b;
    case "*": return a * b;
    case "/": if (b === 0) throw new Error("Division par zéro"); return a / b;
    default: throw new Error("Opérateur invalide");
  }
}
```

Intercepter une erreur avec `try { ... } catch (error) { ... }` pour afficher `error.message`. La fonction ci-dessus est une brique logique, pas toute l’interface.

### Morpion

État initial : `useState(Array(9).fill(null))`, joueur `"X"`. Les cases vont de 0 à 8, ligne par ligne.

```js
function winner(board) {
  const lines = [[0,1,2], [3,4,5], [6,7,8], [0,3,6],
    [1,4,7], [2,5,8], [0,4,8], [2,4,6]];
  for (const [a, b, c] of lines)
    if (board[a] && board[a] === board[b] && board[a] === board[c])
      return board[a];
  return null;
}
```

Au clic : refuser une case occupée ou une partie finie → copier `[...board]` → écrire le joueur dans la copie → `setBoard(next)` → alterner X/O. **Vérifier le gagnant avant le nul** ; sinon un plateau plein gagnant serait mal classé. Pas de gagnant + toutes les cases remplies = nul.

## 13. Erreurs fréquentes et remise

| Problème | Vérifier |
|---|---|
| Page blanche / compilation impossible | Première erreur du terminal et de la console ; imports, accolades, balises fermées. |
| Boucle de rendus | Setter appelé au rendu, `onClick={fn()}`, effet qui modifie sa propre dépendance. |
| L’écran ne change pas | Mutation directe du state, variable ordinaire au lieu de state, mauvais gestionnaire. |
| `23` au lieu de `5` | Convertir les chaînes en nombres. |
| Champ bloqué | `value` sans `onChange`, ou mauvais nom de champ. |
| Mauvais score / message trop tôt | Calcul dérivé du bon tableau ; seuil et limite exacts. |
| Image absente | Import/chemin, extension, casse du fichier. |

**Pour 90 minutes :** 10 min lecture → 10 min structure → 35 min logique → 20 min tests → 10 min dernière vérification → 5 min ZIP/remise.

**Avant d’envoyer :** tester le chemin normal, les seuils exacts, les clics répétés, les limites, la fin de partie et la console. Vérifier que le projet démarre.

Le sujet récent indique `npx create-react-app imagepair`, puis une remise du **dossier `src` compressé, sans les images** : pas `node_modules`, pas `build`, pas tout le projet. Dans un projet CRA déjà préparé : `npm start`. Suivre l’environnement et la consigne effectivement donnés le jour du test.

## Sources

Cours Boostcamp Application Development : Lecture 4 JavaScript, ADLec1/2, Reading API, Lecture 1/2 React, tutoriels, MultipleInputForm et exemples StudentContext/Registration/Summary ; Module Handbook ; Past Midterm ; Application Development Midterm exam publié le 15 octobre 2026. Les règles spécifiques ci-dessus viennent de ces sujets ; le prochain énoncé peut différer.
