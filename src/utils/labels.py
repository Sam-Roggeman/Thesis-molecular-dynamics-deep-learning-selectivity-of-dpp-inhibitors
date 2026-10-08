"""Binding-class vocabulary and ligand-to-class mapping."""

from typing import Final


BindingType = str


def get_binding_classes() -> list[BindingType]:
	"""Return the binding classes in the order expected by the classifiers."""
	return ["apo", "nonbinder", "dpp9selective", "dpp8selective", "aselective"]


ligand_to_binding_type: Final[dict[str, BindingType]] = {
	"12i": "nonbinder",
	"42": "dpp9selective",
	"000808": "dpp8selective",
	"0003822": "aselective",
	"apo": "apo",
	"0000157": "nonbinder",
	"0005356": "dpp9selective",
	"0005862": "dpp8selective",
	"0005362": "aselective",
	"0000193": "nonbinder",
	"0004067": "aselective",
	"0004804": "dpp9selective",
	"0005858": "dpp8selective",
}


def ligant_to_class(ligand_name: str | None) -> BindingType:
	"""Map a ligand identifier to its binding class.

	Args:
		ligand_name: Known ligand identifier, or ``None`` for an apo sample.

	Returns:
		The canonical binding-class label.

	Raises:
		ValueError: If ``ligand_name`` is not in ``ligand_to_binding_type``.
	"""
	if ligand_name is None:
		return "apo"

	ligand_name = ligand_name.lower()
	if ligand_name not in ligand_to_binding_type:
		raise ValueError(f"Unknown ligand name {ligand_name}.")
	return ligand_to_binding_type[ligand_name]


__all__ = ["get_binding_classes", "ligand_to_binding_type", "ligant_to_class"]
